from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from instanseg.utils.models.InstanSeg_UNet import EncoderBlock, Decoder
from instanseg.utils.models.InstanSeg_SAM import LoRALinear, _inject_lora


class InstanSeg_DINO(nn.Module):
    """
    Architecture:
        Block1 (EncoderBlock, no pool): in_channels → 16ch @ full resolution
        DINOv2 (Cellpose trick):        16ch → 1024ch @ 1/ps resolution (patch tokens)
        Neck:                            1024ch → 256ch @ 1/ps resolution
        Decoder (log2(ps) blocks):      256ch → ... → 16ch → dim_out @ full resolution

    Skip connections:
        - DINO intermediates projected + upsampled → skips at intermediate resolutions
        - Block1 output → skip at full resolution
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        layers=(64,),
        norm="BATCH",
        dropout=0,
        act="ReLU",
        dino_model="dinov2_vitl14",
        patch_size=8,
    ):
        super().__init__()

        block1_ch = 16
        neck_ch = 256

        self.block1_ch = block1_ch
        self.ps = patch_size
        n_upsamples = int(math.log2(patch_size))

        # --- Block 1: local features at full resolution ---
        self.block1 = EncoderBlock(in_channels, block1_ch, pool=False,
                                   norm=norm, act=act, dropout=dropout)

        # --- DINOv2 backbone ---
        self.dino = torch.hub.load("facebookresearch/dinov2", dino_model, pretrained=True)
        embed_dim = self.dino.embed_dim  # 1024 for ViT-L
        num_blocks = len(self.dino.blocks)
        self.embed_dim = embed_dim

        # Cellpose trick: replace patch_embed with smaller patches and custom input channels
        self._modify_patch_embed(block1_ch, patch_size)

        # Freeze backbone by default
        for p in self.dino.parameters():
            p.requires_grad = False

        # --- Neck: project transformer features to neck_ch ---
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, neck_ch, 1, bias=False),
            nn.BatchNorm2d(neck_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(neck_ch, neck_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(neck_ch),
            nn.ReLU(inplace=True),
        )

        # --- Decoder layers: [neck_ch, ..intermediates.., block1_ch] ---
        ch = neck_ch
        intermediates = []
        for _ in range(n_upsamples - 1):
            ch = max(ch // 2, block1_ch * 2)
            intermediates.append(ch)
        dec_layers = [neck_ch] + intermediates + [block1_ch]

        # --- Skip connections from DINO intermediates ---
        n_dino_skips = n_upsamples - 1  # last skip comes from block1
        self.extract_indices = [
            (i + 1) * num_blocks // (n_dino_skips + 1) - 1
            for i in range(n_dino_skips)
        ]
        # Projection channels must match decoder expectations after skip reversal:
        # skip_projs[0] has the largest upsample (finest res) → paired with later decoder block
        # skip_projs[-1] has the smallest upsample (coarsest res) → paired with first decoder block
        self.skip_projs = nn.ModuleList([
            nn.Conv2d(embed_dim, dec_layers[n_dino_skips - i], 1)
            for i in range(n_dino_skips)
        ])

        # Multi-head decoders
        if type(out_channels) == int:
            out_channels = [[out_channels]]
        if type(out_channels[0]) == int:
            out_channels = [out_channels]

        self.decoders = nn.ModuleList([
            Decoder(dec_layers, out_ch, norm, act, dropout=dropout)
            for out_ch in out_channels
        ])

        self._has_lora = False
        print(f"InstanSeg_DINO: ps={patch_size}, tokens={256//patch_size}x{256//patch_size}, "
              f"dec_layers={dec_layers}, extract_indices={self.extract_indices}")

    def _modify_patch_embed(self, in_chans, patch_size):
        """Replace DINOv2 patch_embed with Cellpose trick: smaller patches, custom input channels."""
        pe = self.dino.patch_embed
        old_proj = pe.proj  # Conv2d(3, 1024, 14, 14)
        old_w = old_proj.weight.data  # (1024, 3, 14, 14)
        embed_dim = old_w.shape[0]

        # Resample spatial: 14×14 → ps×ps
        new_w = F.interpolate(old_w, size=(patch_size, patch_size),
                              mode="bilinear", align_corners=False)
        # Adjust input channels: 3 → in_chans
        if in_chans != old_w.shape[1]:
            new_w = new_w.mean(dim=1, keepdim=True).repeat(1, in_chans, 1, 1)

        new_proj = nn.Conv2d(in_chans, embed_dim,
                             kernel_size=patch_size, stride=patch_size)
        new_proj.weight = nn.Parameter(new_w)
        new_proj.bias = nn.Parameter(old_proj.bias.data.clone())
        pe.proj = new_proj

        # Update PatchEmbed metadata so its forward() assertions pass
        pe.patch_size = (patch_size, patch_size)
        pe.in_chans = in_chans

    def _interpolate_pos_embed(self, num_patches_h, num_patches_w):
        """Interpolate DINOv2 pos_embed from original grid to new grid size."""
        pos_embed = self.dino.pos_embed  # (1, 1 + N_orig, E)
        cls_pos = pos_embed[:, :1]  # (1, 1, E)
        patch_pos = pos_embed[:, 1:]  # (1, N_orig, E)

        N_orig = patch_pos.shape[1]
        orig_size = int(math.sqrt(N_orig))
        dim = patch_pos.shape[-1]

        if num_patches_h == orig_size and num_patches_w == orig_size:
            return pos_embed

        patch_pos = patch_pos.float().reshape(1, orig_size, orig_size, dim).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(num_patches_h, num_patches_w),
                                  mode="bicubic", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat([cls_pos, patch_pos], dim=1).to(pos_embed.dtype)

    def freeze_backbone(self):
        for param in self.dino.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        if self._has_lora:
            for module in self.dino.modules():
                if isinstance(module, LoRALinear):
                    module.lora_A.requires_grad = True
                    module.lora_B.requires_grad = True
        else:
            for param in self.dino.parameters():
                param.requires_grad = True

    def enable_lora(self, rank: int = 16):
        _inject_lora(self.dino, ("qkv", "proj"), rank=rank)
        self._has_lora = True
        trainable = sum(p.numel() for p in self.dino.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.dino.parameters())
        print(f"LoRA enabled (rank={rank}): {trainable/1e6:.1f}M / {total/1e6:.1f}M DINO params trainable")

    def _run_dino(self, x):
        """Run DINOv2 patch embed + transformer blocks, extracting intermediates."""
        B, C, H, W = x.shape
        nph, npw = H // self.ps, W // self.ps

        # Patch embed: (B, C, H, W) → (B, N, E)
        x = self.dino.patch_embed(x)

        # Prepend CLS token
        cls_tokens = self.dino.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 1+N, E)

        # Add interpolated positional embeddings
        pos = self._interpolate_pos_embed(nph, npw)
        x = x + pos.to(x.dtype)

        # Transformer blocks — extract intermediates
        intermediates = []
        for i, blk in enumerate(self.dino.blocks):
            x = blk(x)
            if i in self.extract_indices:
                mid_patches = x[:, 1:]  # (B, N, E) — skip CLS
                intermediates.append(mid_patches.reshape(B, nph, npw, -1).permute(0, 3, 1, 2))

        # Final norm + extract patch tokens
        x = self.dino.norm(x)
        patch_tokens = x[:, 1:]
        spatial = patch_tokens.reshape(B, nph, npw, -1).permute(0, 3, 1, 2)

        neck_out = self.neck(spatial)
        return intermediates, neck_out

    def forward(self, x):
        block1_out = self.block1(x)  # (B, 16, H, W)

        dino_intermediates, dino_neck = self._run_dino(block1_out)
        # dino_neck: (B, 256, H/ps, W/ps)

        # Build skip connections (collected high-res first, reversed by Decoder)
        # First: block1 at full resolution
        skips = [block1_out]
        # Then: DINO intermediates projected + upsampled (in order of decreasing resolution)
        for i, (feat, proj) in enumerate(zip(dino_intermediates, self.skip_projs)):
            projected = proj(feat)  # (B, ch, H/ps, W/ps)
            scale = 2 ** (len(self.skip_projs) - i)  # later intermediates get less upsampling
            upsampled = F.interpolate(projected, scale_factor=scale,
                                      mode="bilinear", align_corners=False)
            skips.append(upsampled)

        # Decoder reverses skips: [coarsest_dino_skip, ..., block1@full_res]
        return torch.cat([decoder(dino_neck, skips) for decoder in self.decoders], dim=1)


if __name__ == "__main__":
    for ps in [4, 8]:
        print(f"\n--- patch_size={ps} ---")
        model = InstanSeg_DINO(
            in_channels=3,
            out_channels=[[2, 2, 1]],
            layers=[64],
            patch_size=ps,
        )
        model.eval()

        print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

        with torch.no_grad():
            for H, W in [(256, 256), (128, 128), (256, 512)]:
                x = torch.randn(1, 3, H, W)
                y = model(x)
                assert y.shape[-2:] == (H, W), f"Failed for ({H},{W}): got {y.shape}"
                print(f"Input ({H},{W}) → Output {y.shape} ✓")

    print("\nAll tests passed!")
