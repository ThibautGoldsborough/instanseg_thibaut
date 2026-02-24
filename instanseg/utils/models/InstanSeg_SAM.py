from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from instanseg.utils.models.InstanSeg_UNet import EncoderBlock, Decoder


class LoRALinear(nn.Module):
    """Low-rank adaptation wrapper for nn.Linear layers."""

    def __init__(self, original: nn.Linear, rank: int = 16):
        super().__init__()
        self.original = original
        # Freeze the original weights
        for p in self.original.parameters():
            p.requires_grad = False
        self.lora_A = nn.Parameter(torch.randn(original.in_features, rank) * (1 / rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, original.out_features))

    def forward(self, x):
        return self.original(x) + (x @ self.lora_A) @ self.lora_B


def _inject_lora(model: nn.Module, target_names: tuple[str, ...], rank: int = 16):
    """Replace target nn.Linear layers with LoRALinear wrappers in-place."""
    for name, module in model.named_modules():
        for attr in target_names:
            if hasattr(module, attr):
                original = getattr(module, attr)
                if isinstance(original, nn.Linear):
                    setattr(module, attr, LoRALinear(original, rank=rank))


class InstanSeg_SAM(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        layers=(128, 64, 32),
        norm="BATCH",
        dropout=0,
        act="ReLU",
        sam_checkpoint="~/.sam/sam_vit_l_0b3195.pth",
        num_blocks=24,
        window_size=14,
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
        # After encoder: skips are enc_layers[0:-1] channels, bottleneck is enc_layers[-1]
        # e.g. enc_layers=[32,64,128] → skip0=32, skip1=64, bottleneck=128

        # --- SAM input projection ---
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, 3, 3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )

        # --- SAM encoder (ViT-L config) ---
        from segment_anything.modeling.image_encoder import ImageEncoderViT
        self.sam_encoder = ImageEncoderViT(
            img_size=1024,
            patch_size=16,
            in_chans=3,
            embed_dim=1024,
            depth=num_blocks,
            num_heads=16,
            out_chans=256,
            use_abs_pos=False,
            use_rel_pos=True,
            rel_pos_zero_init=True,
            window_size=window_size,
            global_attn_indexes=(5, 11, 17, 23),
        )

        # We extract intermediate features from evenly spaced blocks
        # For layers=[128,64,32], enc has 3 blocks → 2 skips + 1 bottleneck
        num_skips = len(enc_layers) - 1  # number of skip connections
        # Spread extraction points evenly across first 2/3 of blocks
        self.sam_extract_indices = [
            (i + 1) * num_blocks // (num_skips + 1) - 1
            for i in range(num_skips)
        ]
        # e.g. for num_skips=2, num_blocks=24 → indices [7, 15]

        # --- Fusion modules for skip connections ---
        # SAM intermediate features have embed_dim=1024 channels
        self.skip_proj = nn.ModuleList()
        self.skip_fuse = nn.ModuleList()
        for i in range(num_skips):
            skip_ch = enc_layers[i]
            self.skip_proj.append(nn.Conv2d(1024, skip_ch, 1))
            self.skip_fuse.append(nn.Conv2d(skip_ch * 2, skip_ch, 1))

        # --- Fusion module for bottleneck ---
        bn_ch = enc_layers[-1]
        self.bn_proj = nn.Conv2d(256, bn_ch, 1)  # SAM neck outputs 256 channels
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

        # Load pretrained SAM weights
        self._load_sam_weights(sam_checkpoint)

    def _load_sam_weights(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path).expanduser()
        if not checkpoint_path.exists():
            print(f"SAM checkpoint not found at {checkpoint_path}, skipping weight loading.")
            return

        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        # Map keys: checkpoint has "image_encoder.X" → we need "X" for sam_encoder
        sam_state = {}
        for k, v in state_dict.items():
            if k.startswith("image_encoder."):
                new_key = k[len("image_encoder."):]
                if new_key == "pos_embed":
                    continue  # skip absolute positional embedding
                sam_state[new_key] = v

        missing, unexpected = self.sam_encoder.load_state_dict(sam_state, strict=False)
        if missing:
            print(f"SAM missing keys: {missing}")
        if unexpected:
            print(f"SAM unexpected keys: {unexpected}")

    def freeze_backbone(self):
        """Freeze all pretrained SAM weights (encoder blocks, patch_embed, neck)."""
        for param in self.sam_encoder.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all SAM weights (or just LoRA params if LoRA has been injected)."""
        if self._has_lora:
            # Only unfreeze LoRA parameters, keep original weights frozen
            for module in self.sam_encoder.modules():
                if isinstance(module, LoRALinear):
                    module.lora_A.requires_grad = True
                    module.lora_B.requires_grad = True
        else:
            for param in self.sam_encoder.parameters():
                param.requires_grad = True

    def enable_lora(self, rank: int = 16):
        """Inject LoRA adapters into SAM attention layers (qkv and proj)."""
        _inject_lora(self.sam_encoder, ("qkv", "proj"), rank=rank)
        self._has_lora = True
        trainable = sum(p.numel() for p in self.sam_encoder.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.sam_encoder.parameters())
        print(f"LoRA enabled (rank={rank}): {trainable/1e6:.1f}M / {total/1e6:.1f}M SAM params trainable")

    def _run_sam_blocks(self, x):
        """Run SAM patch embed + blocks, extracting intermediate features."""
        x = self.sam_encoder.patch_embed(x)
        # x: (B, H/16, W/16, 1024)

        intermediates = []
        for i, blk in enumerate(self.sam_encoder.blocks):
            x = blk(x)
            if i in self.sam_extract_indices:
                # (B, H', W', C) → (B, C, H', W')
                intermediates.append(x.permute(0, 3, 1, 2))

        # Apply neck: permute to (B, C, H', W') first
        neck_out = self.sam_encoder.neck(x.permute(0, 3, 1, 2))
        return intermediates, neck_out

    def forward(self, x):
        # --- UNet encoder path ---
        skips = []
        h = x
        for n, layer in enumerate(self.encoder):
            h = layer(h)
            if n < len(self.encoder) - 1:
                skips.append(h)
        bottleneck = h

        # --- SAM path ---
        sam_input = self.input_proj(x)
        sam_intermediates, sam_neck = self._run_sam_blocks(sam_input)

        # --- Fuse skip connections ---
        for i in range(len(skips)):
            skip = skips[i]
            sam_feat = sam_intermediates[i]
            # Resize SAM feature to match skip spatial dims
            sam_feat = F.interpolate(sam_feat, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            sam_feat = self.skip_proj[i](sam_feat)
            skips[i] = self.skip_fuse[i](torch.cat([skip, sam_feat], dim=1))

        # --- Fuse bottleneck ---
        sam_bn = F.interpolate(sam_neck, size=bottleneck.shape[-2:], mode="bilinear", align_corners=False)
        sam_bn = self.bn_proj(sam_bn)
        bottleneck = self.bn_fuse(torch.cat([bottleneck, sam_bn], dim=1))

        # --- Decode ---
        return torch.cat([decoder(bottleneck, skips) for decoder in self.decoders], dim=1)


if __name__ == "__main__":
    model = InstanSeg_SAM(
        in_channels=3,
        out_channels=[[2, 2, 1]],
        layers=[128, 64, 32],
        sam_checkpoint="nonexistent",  # skip loading pretrained weights for testing
    )
    model.eval()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    with torch.no_grad():
        for H, W in [(256, 256), (128, 128), (384, 384), (256, 512), (160, 320)]:
            x = torch.randn(1, 3, H, W)
            y = model(x)
            assert y.shape[-2:] == (H, W), f"Failed for ({H},{W}): got {y.shape}"
            print(f"Input ({H},{W}) → Output {y.shape} ✓")

    print("\nAll tests passed!")
