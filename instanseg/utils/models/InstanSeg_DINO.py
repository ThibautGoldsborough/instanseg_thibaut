from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from instanseg.utils.models.InstanSeg_UNet import EncoderBlock, Decoder
from instanseg.utils.models.InstanSeg_SAM import LoRALinear, _inject_lora


class InstanSeg_DINO(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        layers=(128, 64, 32),
        norm="BATCH",
        dropout=0,
        act="ReLU",
        dino_model="dinov2_vitl14",
    ):
        super().__init__()

        layers = list(layers)
        # layers comes in descending order e.g. [128, 64, 32]
        # encoder needs ascending: [32, 64, 128]
        enc_layers = layers[::-1]

        # --- UNet Encoder ---
        self.encoder = nn.ModuleList(
            [EncoderBlock(in_channels, enc_layers[0], pool=False, norm=norm, act=act, dropout=dropout)]
            + [EncoderBlock(enc_layers[i], enc_layers[i + 1], norm=norm, act=act, dropout=dropout)
               for i in range(len(enc_layers) - 1)]
        )

        # --- DINO input projection ---
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, 3, 3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )

        # --- DINOv2 encoder ---
        self.dino_encoder = torch.hub.load(
            "facebookresearch/dinov2", dino_model, pretrained=True
        )

        embed_dim = self.dino_encoder.embed_dim  # 1024 for ViT-L

        # Extraction indices for intermediate features
        num_blocks = len(self.dino_encoder.blocks)
        num_skips = len(enc_layers) - 1
        self.dino_extract_indices = [
            (i + 1) * num_blocks // (num_skips + 1) - 1
            for i in range(num_skips)
        ]
        # e.g. for num_skips=2, num_blocks=24 → indices [7, 15]

        # --- Fusion modules for skip connections ---
        self.skip_proj = nn.ModuleList()
        self.skip_fuse = nn.ModuleList()
        for i in range(num_skips):
            skip_ch = enc_layers[i]
            self.skip_proj.append(nn.Conv2d(embed_dim, skip_ch, 1))
            self.skip_fuse.append(nn.Conv2d(skip_ch * 2, skip_ch, 1))

        # --- Fusion module for bottleneck ---
        bn_ch = enc_layers[-1]
        self.bn_proj = nn.Conv2d(embed_dim, bn_ch, 1)
        self.bn_fuse = nn.Conv2d(bn_ch * 2, bn_ch, 1)

        # --- Decoder ---
        if type(out_channels) == int:
            out_channels = [[out_channels]]
        if type(out_channels[0]) == int:
            out_channels = [out_channels]

        self.decoders = nn.ModuleList(
            [Decoder(layers, out_channel, norm, act, dropout=dropout) for out_channel in out_channels]
        )

        self._has_lora = False

    def freeze_backbone(self):
        """Freeze all pretrained DINOv2 weights."""
        for param in self.dino_encoder.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all DINOv2 weights (or just LoRA params if LoRA has been injected)."""
        if self._has_lora:
            for module in self.dino_encoder.modules():
                if isinstance(module, LoRALinear):
                    module.lora_A.requires_grad = True
                    module.lora_B.requires_grad = True
        else:
            for param in self.dino_encoder.parameters():
                param.requires_grad = True

    def enable_lora(self, rank: int = 16):
        """Inject LoRA adapters into DINOv2 attention layers (qkv and proj)."""
        _inject_lora(self.dino_encoder, ("qkv", "proj"), rank=rank)
        self._has_lora = True
        trainable = sum(p.numel() for p in self.dino_encoder.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.dino_encoder.parameters())
        print(f"LoRA enabled (rank={rank}): {trainable/1e6:.1f}M / {total/1e6:.1f}M DINO params trainable")

    def _run_dino_blocks(self, x):
        """Run DINOv2 patch embed + blocks, extracting intermediate features."""
        B, C, H, W = x.shape
        patch_size = self.dino_encoder.patch_size

        # Pad to next multiple of patch_size so patch_embed doesn't fail
        pad_h = (patch_size - H % patch_size) % patch_size
        pad_w = (patch_size - W % patch_size) % patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        H_pad, W_pad = x.shape[-2:]

        # Patch embedding
        tokens = self.dino_encoder.patch_embed(x)
        # tokens: (B, N, embed_dim) where N = (H_pad/patch_size) * (W_pad/patch_size)

        # Prepend CLS token
        cls_token = self.dino_encoder.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_token, tokens], dim=1)

        # Add interpolated positional embeddings
        # DINOv2 convention: interpolate_pos_encoding(x, w, h) where w=height, h=width
        tokens = tokens + self.dino_encoder.interpolate_pos_encoding(tokens, H_pad, W_pad)

        h_patches = H_pad // patch_size
        w_patches = W_pad // patch_size

        intermediates = []
        for i, blk in enumerate(self.dino_encoder.blocks):
            tokens = blk(tokens)
            if i in self.dino_extract_indices:
                # Strip CLS token, reshape to spatial
                patch_tokens = tokens[:, 1:, :]  # (B, N, embed_dim)
                feat = patch_tokens.permute(0, 2, 1).reshape(B, -1, h_patches, w_patches)
                intermediates.append(feat)

        # Final norm + bottleneck features
        tokens = self.dino_encoder.norm(tokens)
        patch_tokens = tokens[:, 1:, :]
        bottleneck_feat = patch_tokens.permute(0, 2, 1).reshape(B, -1, h_patches, w_patches)

        return intermediates, bottleneck_feat

    def forward(self, x):
        # --- UNet encoder path ---
        skips = []
        h = x
        for n, layer in enumerate(self.encoder):
            h = layer(h)
            if n < len(self.encoder) - 1:
                skips.append(h)
        bottleneck = h

        # --- DINO path ---
        dino_input = self.input_proj(x)
        dino_intermediates, dino_bottleneck = self._run_dino_blocks(dino_input)

        # --- Fuse skip connections ---
        for i in range(len(skips)):
            skip = skips[i]
            dino_feat = dino_intermediates[i]
            dino_feat = F.interpolate(dino_feat, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            dino_feat = self.skip_proj[i](dino_feat)
            skips[i] = self.skip_fuse[i](torch.cat([skip, dino_feat], dim=1))

        # --- Fuse bottleneck ---
        dino_bn = F.interpolate(dino_bottleneck, size=bottleneck.shape[-2:], mode="bilinear", align_corners=False)
        dino_bn = self.bn_proj(dino_bn)
        bottleneck = self.bn_fuse(torch.cat([bottleneck, dino_bn], dim=1))

        # --- Decode ---
        return torch.cat([decoder(bottleneck, skips) for decoder in self.decoders], dim=1)


if __name__ == "__main__":
    model = InstanSeg_DINO(
        in_channels=3,
        out_channels=[[2, 2, 1]],
        layers=[128, 64, 32],
    )
    model.eval()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    with torch.no_grad():
        for H, W in [(224, 224), (256, 256), (280, 280), (392, 392), (280, 560), (168, 336)]:
            x = torch.randn(1, 3, H, W)
            y = model(x)
            assert y.shape[-2:] == (H, W), f"Failed for ({H},{W}): got {y.shape}"
            print(f"Input ({H},{W}) → Output {y.shape} ✓")

    print("\nAll tests passed!")
