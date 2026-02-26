from __future__ import annotations
import math
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
        for p in self.original.parameters():
            p.requires_grad = False
        device = original.weight.device
        self.lora_A = nn.Parameter(torch.randn(original.in_features, rank, device=device) * (1 / rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, original.out_features, device=device))

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
    """
    Architecture:
        Block1 (EncoderBlock, no pool): in_channels → 16ch @ full resolution
        SAM (Cellpose trick):           16ch → 256ch @ 1/ps resolution
        Decoder (log2(ps) blocks):      256ch → ... → 16ch → dim_out @ full resolution

    Skip connections:
        - SAM intermediates projected + upsampled → skips at intermediate resolutions
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
        sam_checkpoint="~/.sam/sam_vit_l_0b3195.pth",
        num_blocks=24,
        patch_size=8,
    ):
        super().__init__()

        block1_ch = 16
        sam_neck_ch = 256
        sam_embed_dim = 1024

        self.block1_ch = block1_ch
        self.sam_embed_dim = sam_embed_dim
        self.ps = patch_size
        n_upsamples = int(math.log2(patch_size))

        # --- Block 1: local features at full resolution ---
        self.block1 = EncoderBlock(in_channels, block1_ch, pool=False,
                                   norm=norm, act=act, dropout=dropout)

        # --- SAM encoder (Cellpose trick: custom ps, all blocks global) ---
        from segment_anything.modeling.image_encoder import ImageEncoderViT
        self.sam_encoder = ImageEncoderViT(
            img_size=256,
            patch_size=patch_size,
            in_chans=block1_ch,
            embed_dim=sam_embed_dim,
            depth=num_blocks,
            num_heads=16,
            out_chans=sam_neck_ch,
            use_abs_pos=True,
            use_rel_pos=True,
            rel_pos_zero_init=True,
            window_size=14,
            global_attn_indexes=list(range(num_blocks)),
        )

        # --- Decoder layers: [neck_ch, ..intermediates.., block1_ch] ---
        ch = sam_neck_ch
        intermediates = []
        for _ in range(n_upsamples - 1):
            ch = max(ch // 2, block1_ch * 2)
            intermediates.append(ch)
        dec_layers = [sam_neck_ch] + intermediates + [block1_ch]

        # --- Skip connections from SAM intermediates ---
        n_sam_skips = n_upsamples - 1  # last skip comes from block1
        self.sam_extract_indices = [
            (i + 1) * num_blocks // (n_sam_skips + 1) - 1
            for i in range(n_sam_skips)
        ]
        # Projection channels must match decoder expectations after skip reversal:
        # skip_projs[0] has the largest upsample (finest res) → paired with later decoder block
        # skip_projs[-1] has the smallest upsample (coarsest res) → paired with first decoder block
        self.skip_projs = nn.ModuleList([
            nn.Conv2d(sam_embed_dim, dec_layers[n_sam_skips - i], 1)
            for i in range(n_sam_skips)
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
        self._load_sam_weights(sam_checkpoint, block1_ch, patch_size)

        print(f"InstanSeg_SAM: ps={patch_size}, tokens={256//patch_size}x{256//patch_size}, "
              f"dec_layers={dec_layers}, extract_indices={self.sam_extract_indices}")

    def _load_sam_weights(self, checkpoint_path, in_chans, patch_size):
        """Load pretrained SAM-ViT-L weights with Cellpose trick resampling."""
        checkpoint_path = Path(checkpoint_path).expanduser()
        if not checkpoint_path.exists():
            print(f"SAM checkpoint not found at {checkpoint_path}, skipping weight loading.")
            return

        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        sam_state = {}
        for k, v in state_dict.items():
            if k.startswith("image_encoder."):
                sam_state[k[len("image_encoder."):]] = v

        # Resample patch_embed kernel: (1024, 3, 16, 16) → (1024, in_chans, ps, ps)
        key = "patch_embed.proj.weight"
        if key in sam_state:
            old_w = sam_state[key]  # (1024, 3, 16, 16)
            new_w = F.interpolate(old_w, size=(patch_size, patch_size),
                                  mode="bilinear", align_corners=False)
            if in_chans != old_w.shape[1]:
                new_w = new_w.mean(dim=1, keepdim=True).repeat(1, in_chans, 1, 1)
            sam_state[key] = new_w

        # Resample pos_embed if spatial dimensions differ
        pe_key = "pos_embed"
        if pe_key in sam_state:
            old_pe = sam_state[pe_key]  # (1, H_orig, W_orig, E)
            expected_shape = self.sam_encoder.pos_embed.shape
            if old_pe.shape != expected_shape:
                old_pe = old_pe.permute(0, 3, 1, 2)  # (1, E, H, W)
                old_pe = F.interpolate(old_pe, size=expected_shape[1:3],
                                       mode="bicubic", align_corners=False)
                sam_state[pe_key] = old_pe.permute(0, 2, 3, 1)

        # Resample rel_pos for windowed→global blocks (and any size mismatches)
        own_state = self.sam_encoder.state_dict()
        for k, v in list(sam_state.items()):
            if "rel_pos" in k and k in own_state:
                expected_len = own_state[k].shape[0]
                if v.shape[0] != expected_len:
                    sam_state[k] = F.interpolate(
                        v.unsqueeze(0).permute(0, 2, 1),
                        size=expected_len,
                        mode="linear",
                        align_corners=False,
                    ).permute(0, 2, 1).squeeze(0)

        missing, unexpected = self.sam_encoder.load_state_dict(sam_state, strict=False)
        if missing:
            print(f"SAM missing keys: {missing}")
        if unexpected:
            print(f"SAM unexpected keys: {unexpected}")

    def freeze_backbone(self):
        for param in self.sam_encoder.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        if self._has_lora:
            for module in self.sam_encoder.modules():
                if isinstance(module, LoRALinear):
                    module.lora_A.requires_grad = True
                    module.lora_B.requires_grad = True
        else:
            for param in self.sam_encoder.parameters():
                param.requires_grad = True

    def enable_lora(self, rank: int = 16):
        _inject_lora(self.sam_encoder, ("qkv", "proj"), rank=rank)
        self._has_lora = True
        trainable = sum(p.numel() for p in self.sam_encoder.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.sam_encoder.parameters())
        print(f"LoRA enabled (rank={rank}): {trainable/1e6:.1f}M / {total/1e6:.1f}M SAM params trainable")

    def _run_sam(self, x):
        """Run SAM patch embed + transformer blocks + neck, extracting intermediates."""
        x = self.sam_encoder.patch_embed(x)
        # x: (B, H/ps, W/ps, embed_dim) — SAM uses BHWC format internally

        if self.sam_encoder.pos_embed is not None:
            pos = self.sam_encoder.pos_embed
            if x.shape[1:3] != pos.shape[1:3]:
                pos = pos.permute(0, 3, 1, 2)
                pos = F.interpolate(pos, size=x.shape[1:3],
                                    mode="bicubic", align_corners=False)
                pos = pos.permute(0, 2, 3, 1)
            x = x + pos

        intermediates = []
        for i, blk in enumerate(self.sam_encoder.blocks):
            x = blk(x)
            if i in self.sam_extract_indices:
                intermediates.append(x.permute(0, 3, 1, 2))  # BHWC → BCHW

        neck_out = self.sam_encoder.neck(x.permute(0, 3, 1, 2))
        return intermediates, neck_out

    def forward(self, x):
        block1_out = self.block1(x)  # (B, 16, H, W)

        sam_intermediates, sam_neck = self._run_sam(block1_out)
        # sam_neck: (B, 256, H/ps, W/ps)

        # Build skip connections (collected high-res first, reversed by Decoder)
        skips = [block1_out]
        for i, (feat, proj) in enumerate(zip(sam_intermediates, self.skip_projs)):
            projected = proj(feat)
            scale = 2 ** (len(self.skip_projs) - i)
            upsampled = F.interpolate(projected, scale_factor=scale,
                                      mode="bilinear", align_corners=False)
            skips.append(upsampled)

        return torch.cat([decoder(sam_neck, skips) for decoder in self.decoders], dim=1)


if __name__ == "__main__":
    for ps in [4, 8]:
        print(f"\n--- patch_size={ps} ---")
        model = InstanSeg_SAM(
            in_channels=3,
            out_channels=[[2, 2, 1]],
            layers=[64],
            sam_checkpoint="nonexistent",
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
