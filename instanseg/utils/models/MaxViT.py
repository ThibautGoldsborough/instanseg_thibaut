from __future__ import annotations

import dataclasses
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from instanseg.utils.models.InstanSeg_UNet import conv_norm_act


# Preset -> (timm arch name, default training resolution used to fetch pretrained).
# We pick the `*_rmlp_*_rw_*` variants where possible: relative MLP position bias
# is resolution-agnostic, so the pretrained weights transfer cleanly to 256x256
# inputs. `large` falls back to a `_tf_*` variant (no rmlp equivalent exists) —
# its window-based rel_pos_bias tables are interpolated by timm at load time.
_PRESETS: dict[str, dict] = {
    "tiny":  dict(timm_name="maxvit_rmlp_pico_rw_256"),
    "base":  dict(timm_name="maxvit_rmlp_tiny_rw_256"),
    "large": dict(timm_name="maxvit_large_tf_512"),
}


def _cfg_base_name(timm_name: str) -> str:
    """Strip the resolution suffix (_256, _384, _512, _224) used in create_model
    to get the config key under timm.models.maxxvit.model_cfgs."""
    for suffix in ("_256", "_384", "_512", "_224"):
        if timm_name.endswith(suffix):
            return timm_name[: -len(suffix)]
    return timm_name


def _build_timm_maxvit(timm_name: str, in_channels: int, img_size: int,
                       pretrained: bool, drop_path_rate: float) -> nn.Module:
    """Create a timm MaxxVit and rebuild it at our img_size.

    We stay in `features_only=False` mode so `.stem` / `.stages` are reachable
    for weight copying; we'll extract multi-scale features by running them
    manually in `MaxViT._run`. timm applies a linear stochastic-depth schedule
    (0 at the first encoder block, ``drop_path_rate`` at the last) internally.
    """
    import timm
    kwargs = dict(
        pretrained=pretrained,
        in_chans=in_channels,
        num_classes=0,
        global_pool="",
        drop_path_rate=drop_path_rate,
    )
    # tf_* variants hard-code window_size at training res; pass img_size to rebuild.
    if "_tf_" in timm_name:
        kwargs["img_size"] = (img_size, img_size)
    return timm.create_model(timm_name, **kwargs)


class ConvBlock(nn.Module):
    """Small residual conv block used at full- and half-res (no attention)."""

    def __init__(self, in_dim: int, out_dim: int,
                 norm: str = "BATCH", act: str = "ReLU", dropout: float = 0.0):
        super().__init__()
        self.shortcut = conv_norm_act(in_dim, out_dim, 1, norm, act)
        self.conv1 = conv_norm_act(in_dim, out_dim, 3, norm, act)
        self.conv2 = conv_norm_act(out_dim, out_dim, 3, norm, act)
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sc = self.shortcut(x)
        x = self.conv1(x)
        return sc + self.dropout(self.conv2(x))


class _TimmDecoderStage(nn.Module):
    """Upsample -> concat skip -> 1x1 fuse -> timm MaxxVitBlocks (stride=1)."""

    def __init__(
        self,
        in_dim: int,
        skip_dim: int,
        out_dim: int,
        depth: int,
        transformer_cfg,
        conv_cfg,
        feat_size: tuple[int, int],
        drop_path: float = 0.0,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        from timm.models.maxxvit import MaxxVitStage

        self.gradient_checkpointing = gradient_checkpointing
        self.fuse = nn.Sequential(
            nn.Conv2d(in_dim + skip_dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
        )
        # A stride=1 MaxxVitStage with in==out channels is structurally identical
        # to the refinement portion of an encoder stage — this is what makes the
        # warm-start via state_dict copy work.
        self.stage = MaxxVitStage(
            in_chs=out_dim, out_chs=out_dim, stride=1, depth=depth,
            feat_size=feat_size, block_types=("M",),
            transformer_cfg=transformer_cfg, conv_cfg=conv_cfg,
            drop_path=[drop_path] * depth,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        if self.gradient_checkpointing and self.training:
            for blk in self.stage.blocks:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            return x
        return self.stage(x)


class _ConvDecoderStage(nn.Module):
    """Upsample + fuse + conv block; used at 1/2 resolution where attention is wasteful."""

    def __init__(self, in_dim: int, skip_dim: int, out_dim: int,
                 norm: str, act: str, dropout: float):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_dim + skip_dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
        )
        self.block = ConvBlock(out_dim, out_dim, norm=norm, act=act, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        return self.block(x)


class _MaxViTDecoder(nn.Module):
    """Full decoder: 3 MaxxVit stages (1/16, 1/8, 1/4) + 1 conv stage (1/2)
    + final 1x1 heads after a 2x upsample back to 1/1."""

    def __init__(
        self,
        dims: Sequence[int],          # [d0, d1, d2, d3]
        stem_dim: int,
        dec_depths: Sequence[int],    # 3 entries for the 3 MaxxVit decoder stages
        out_channels: list[int],
        transformer_cfg,
        conv_cfg,
        img_size: int,
        norm: str,
        act: str,
        dropout: float,
        gradient_checkpointing: bool,
        drop_path: float = 0.0,
    ):
        super().__init__()
        d0, d1, d2, d3 = dims

        self.up3 = _TimmDecoderStage(
            in_dim=d3, skip_dim=d2, out_dim=d2, depth=dec_depths[2],
            transformer_cfg=transformer_cfg, conv_cfg=conv_cfg,
            feat_size=(img_size // 16, img_size // 16),
            drop_path=drop_path,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.up2 = _TimmDecoderStage(
            in_dim=d2, skip_dim=d1, out_dim=d1, depth=dec_depths[1],
            transformer_cfg=transformer_cfg, conv_cfg=conv_cfg,
            feat_size=(img_size // 8, img_size // 8),
            drop_path=drop_path,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.up1 = _TimmDecoderStage(
            in_dim=d1, skip_dim=d0, out_dim=d0, depth=dec_depths[0],
            transformer_cfg=transformer_cfg, conv_cfg=conv_cfg,
            feat_size=(img_size // 4, img_size // 4),
            drop_path=drop_path,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.up0 = _ConvDecoderStage(d0, stem_dim, stem_dim, norm, act, dropout)
        self.final_conv = ConvBlock(stem_dim, stem_dim, norm=norm, act=act, dropout=dropout)

        final_norm = norm if (norm is not None and norm.lower() != "instance") else None
        self.heads = nn.ModuleList(
            [conv_norm_act(stem_dim, oc, 1, norm=final_norm, act=None) for oc in out_channels]
        )

    def forward(self, bottleneck: torch.Tensor, skips: list[torch.Tensor], input_size) -> torch.Tensor:
        # skips = [stem @ 1/2, s0 @ 1/4, s1 @ 1/8, s2 @ 1/16]
        x = self.up3(bottleneck, skips[3])   # -> 1/16, d2
        x = self.up2(x,          skips[2])   # -> 1/8,  d1
        x = self.up1(x,          skips[1])   # -> 1/4,  d0
        x = self.up0(x,          skips[0])   # -> 1/2,  stem_dim
        x = F.interpolate(x, size=input_size, mode="nearest")
        x = self.final_conv(x)
        return torch.cat([h(x) for h in self.heads], dim=1)


class MaxViT(nn.Module):
    """Pretrained timm MaxViT encoder + symmetric UNet-style decoder.

    Encoder: timm's `MaxxVit`. We extract 5 feature levels (stem @ 1/2 and the
    four stage outputs at 1/4, 1/8, 1/16, 1/32).

    Decoder: 3 timm `MaxxVitStage`s (stride=1) mirroring encoder stages 0..2
    at 1/4, 1/8, 1/16, plus a conv stage at 1/2 and a final 2x upsample to 1/1.
    With `init_decoder_from_encoder=True` (default when `pretrained=True`),
    each decoder stage's MaxxVit blocks are warm-started from the encoder
    stage's refinement blocks (blocks[1:]), which have matching shapes. Fuse
    convs and the final conv block stay randomly initialized.

    Decoder depths default to `max(1, encoder_depth - 1)` so every decoder
    block has a matching-shape encoder refinement block to copy from.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels,
        timm_name: str = "maxvit_rmlp_tiny_rw_256",
        pretrained: bool = True,
        img_size: int = 256,
        dec_depths: Optional[Sequence[int]] = None,
        dropout: float = 0.0,
        attn_dropout: Optional[float] = None,
        norm: str = "BATCH",
        act: str = "ReLU",
        amp: bool = False,
        channels_last: bool = True,
        compile: bool = False,
        compile_mode: str = "default",
        amp_dtype: torch.dtype = torch.bfloat16,
        gradient_checkpointing: bool = False,
        init_decoder_from_encoder: Optional[bool] = None,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()

        if attn_dropout is None:
            attn_dropout = dropout
        if init_decoder_from_encoder is None:
            init_decoder_from_encoder = pretrained

        from timm.models.maxxvit import model_cfgs

        self.timm_name = timm_name
        self.img_size = int(img_size)
        self.drop_path_rate = float(drop_path_rate)

        # --- Encoder ---------------------------------------------------------
        self.encoder = _build_timm_maxvit(
            timm_name, in_channels=in_channels, img_size=self.img_size,
            pretrained=pretrained, drop_path_rate=self.drop_path_rate,
        )

        # Grab the config so we can replicate it exactly in the decoder stages.
        cfg = model_cfgs[_cfg_base_name(timm_name)]
        dims = list(cfg.embed_dim)                # [d0, d1, d2, d3]
        enc_depths = list(cfg.depths)             # e.g. (2, 2, 5, 2)
        # stem_width may be int or (conv1_out, conv2_out); the stem output is the last value.
        stem_dim = cfg.stem_width[-1] if isinstance(cfg.stem_width, (tuple, list)) else cfg.stem_width
        stem_dim = int(stem_dim)
        # Propagate window/grid size from img_size through the cfg, mirroring
        # what timm's MaxxVit.__init__ does for the encoder. If the encoder is
        # built at a different img_size, its own stages will have been rebuilt
        # by timm with these same values.
        ws = self.img_size // cfg.transformer_cfg.partition_ratio
        transformer_cfg = dataclasses.replace(
            cfg.transformer_cfg, window_size=(ws, ws), grid_size=(ws, ws)
        )
        # Inputs to forward() must be divisible by (partition_ratio * window_size)
        # in both H and W, since the deepest stage runs at 1/partition_ratio
        # resolution and partitions into ws-sized windows. We pad to this
        # multiple in forward() and crop the output back, so non-multiple and
        # non-square inputs both work.
        self._pad_multiple = int(cfg.transformer_cfg.partition_ratio * ws)
        conv_cfg = cfg.conv_cfg

        if dec_depths is None:
            # One shorter than encoder so every decoder block maps to an encoder
            # refinement block, giving a clean warm-start.
            dec_depths = [max(1, d - 1) for d in enc_depths[:3]]
        dec_depths = [int(x) for x in dec_depths]

        # --- Decoder heads ---------------------------------------------------
        if isinstance(out_channels, int):
            out_channels = [[out_channels]]
        if isinstance(out_channels[0], int):
            out_channels = [out_channels]

        self.decoders = nn.ModuleList(
            [
                _MaxViTDecoder(
                    dims=dims, stem_dim=stem_dim,
                    dec_depths=dec_depths, out_channels=oc,
                    transformer_cfg=transformer_cfg, conv_cfg=conv_cfg,
                    img_size=self.img_size,
                    norm=norm, act=act, dropout=dropout,
                    gradient_checkpointing=gradient_checkpointing,
                    drop_path=self.drop_path_rate,
                )
                for oc in out_channels
            ]
        )

        # Override timm's default linear schedule on the encoder so every block
        # uses the same fixed rate, as in the MaxViT paper's description.
        if self.drop_path_rate > 0:
            from timm.layers import DropPath
            for m in self.encoder.modules():
                if isinstance(m, DropPath):
                    m.drop_prob = float(self.drop_path_rate)

        self._enc_depths = enc_depths
        self._dec_depths = dec_depths

        if init_decoder_from_encoder:
            self._warm_start_decoder_from_encoder()

        # --- Runtime flags (amp / channels_last / compile) -------------------
        self._amp = amp
        self._amp_dtype = amp_dtype
        self._channels_last = channels_last
        if channels_last:
            self.to(memory_format=torch.channels_last)
        self._core = torch.compile(self._run, mode=compile_mode) if compile else self._run

    # ------------------------------------------------------------------ warm-start
    def _warm_start_decoder_from_encoder(self) -> None:
        """Copy encoder refinement-block weights into decoder blocks.

        Encoder stage i has `enc_depths[i]` blocks: block 0 is the downsample
        block (different in_chs), blocks 1..N-1 are refinement blocks
        (in_chs == out_chs == dims[i], stride=1). Decoder stages are built with
        stride=1 and in==out channels, so their blocks match those refinement
        blocks key-for-key. We copy as many as we have slots for.
        """
        report = {}
        for dec in self.decoders:
            for stage_idx, dec_stage in zip((0, 1, 2), (dec.up1, dec.up2, dec.up3)):
                enc_stage = self.encoder.stages[stage_idx]
                refinement = enc_stage.blocks[1:]  # skip the downsample block
                dst_blocks = dec_stage.stage.blocks
                n = min(len(dst_blocks), len(refinement))
                for j in range(n):
                    dst_blocks[j].load_state_dict(refinement[j].state_dict(), strict=True)
                report[f"enc_stage[{stage_idx}]->dec_up{stage_idx+1}"] = f"{n}/{len(dst_blocks)} blocks"
        self._warm_start_report = report

    # ------------------------------------------------------------------ introspection
    def get_embedding_tap(self) -> nn.Module:
        return self.encoder.stages[-1]

    # ------------------------------------------------------------------ forward
    def _run(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        # Extract the 5-level feature pyramid manually so we can use `.stages`
        # directly (needed so warm-start / gc-through-encoder behave predictably).
        stem = self.encoder.stem(x)                    # 1/2,  stem_dim
        s0 = self.encoder.stages[0](stem)              # 1/4,  d0
        s1 = self.encoder.stages[1](s0)                # 1/8,  d1
        s2 = self.encoder.stages[2](s1)                # 1/16, d2
        s3 = self.encoder.stages[3](s2)                # 1/32, d3 (bottleneck)
        skips = [stem, s0, s1, s2]
        return torch.cat([dec(s3, skips, input_size) for dec in self.decoders], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept arbitrary (H, W), square or not, by replicate-padding up to
        # the next multiple of self._pad_multiple in each dim (Swin/SwinIR
        # "pad-and-crop" recipe, e.g. https://arxiv.org/abs/2108.10257). For
        # the _rmlp_ presets this is bit-exact for inputs already on the grid
        # because position bias is an MLP over relative coords (log-CPB,
        # https://arxiv.org/abs/2111.09883) and so is resolution-agnostic.
        H, W = x.shape[-2:]
        m = self._pad_multiple
        pad_h = (-H) % m
        pad_w = (-W) % m
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        if self._channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        if self._amp and x.is_cuda:
            in_dtype = x.dtype
            with torch.autocast("cuda", dtype=self._amp_dtype):
                y = self._core(x)
            y = y.to(in_dtype)
        else:
            y = self._core(x)

        if pad_h or pad_w:
            y = y[..., :H, :W]
        return y


# ---------------------------------------------------------------------------- presets
def _make_preset(name: str, in_channels: int, out_channels, **overrides) -> MaxViT:
    cfg = dict(_PRESETS[name])
    cfg.update(overrides)
    return MaxViT(in_channels=in_channels, out_channels=out_channels, **cfg)


def maxvit_tiny(in_channels: int, out_channels, **kwargs) -> MaxViT:
    return _make_preset("tiny", in_channels, out_channels, **kwargs)


def maxvit_base(in_channels: int, out_channels, **kwargs) -> MaxViT:
    return _make_preset("base", in_channels, out_channels, **kwargs)


def maxvit_large(in_channels: int, out_channels, **kwargs) -> MaxViT:
    # Large needs gradient checkpointing — without it, activations at 1024 ch
    # with MBConv x4 expansion blow past typical GPU memory.
    kwargs.setdefault("gradient_checkpointing", True)
    return _make_preset("large", in_channels, out_channels, **kwargs)


if __name__ == "__main__":
    # No network: exercise architecture with pretrained=False so the test runs
    # offline. Warm-start is skipped automatically in this mode.
    for name, builder in [("tiny", maxvit_tiny), ("base", maxvit_base), ("large", maxvit_large)]:
        print(f"--- maxvit_{name} ---")
        m = builder(in_channels=3, out_channels=[[2, 2, 1]], pretrained=False)
        p = sum(x.numel() for x in m.parameters()) / 1e6
        print(f"  params: {p:.2f}M  (timm={_PRESETS[name]['timm_name']})")
        m.eval()
        # Encoder window/grid sizes are fixed at build time from img_size, so
        # inputs must be multiples of 32*window = 256 for the bottleneck feature
        # to be partitionable. train.py already operates on 256x256 crops.
        with torch.no_grad():
            y = m(torch.randn(1, 3, 256, 256))
        assert y.shape[-2:] == (256, 256), f"Bad output shape: {y.shape}"
        print(f"  input (256,256) -> {tuple(y.shape)}")
