from __future__ import annotations

from tkinter import FALSE
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from instanseg.utils.models.InstanSeg_UNet import conv_norm_act


def _num_heads(dim: int, head_dim: int = 32) -> int:
    return max(1, dim // head_dim)


class SqueezeExcite(nn.Module):
    def __init__(self, dim: int, rd_ratio: float = 0.25):
        super().__init__()
        hidden = max(1, int(dim * rd_ratio))
        self.fc1 = nn.Conv2d(dim, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3), keepdim=True)
        s = F.silu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


class MBConv(nn.Module):
    """Inverted-residual MBConv with SE; optional stride-2 downsample."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        stride: int = 1,
        expansion: float = 4.0,
        se_ratio: float = 0.25,
    ):
        super().__init__()
        hidden = int(in_dim * expansion)
        self.use_residual = (stride == 1 and in_dim == out_dim)

        self.pre_norm = nn.BatchNorm2d(in_dim)
        self.expand = nn.Conv2d(in_dim, hidden, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.dwconv = nn.Conv2d(
            hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False
        )
        self.bn2 = nn.BatchNorm2d(hidden)
        self.se = SqueezeExcite(hidden, se_ratio)
        self.project = nn.Conv2d(hidden, out_dim, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_dim)

        if self.use_residual:
            self.shortcut: nn.Module = nn.Identity()
        else:
            layers: list[nn.Module] = []
            if stride > 1:
                layers.append(nn.AvgPool2d(stride, stride))
            layers += [
                nn.Conv2d(in_dim, out_dim, 1, bias=False),
                nn.BatchNorm2d(out_dim),
            ]
            self.shortcut = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sc = self.shortcut(x)
        x = self.pre_norm(x)
        x = F.silu(self.bn1(self.expand(x)), inplace=True)
        x = F.silu(self.bn2(self.dwconv(x)), inplace=True)
        x = self.se(x)
        x = self.bn3(self.project(x))
        return x + sc


class PartitionAttention(nn.Module):
    """Pre-norm multi-head self-attention + MLP over a block or grid partition.

    Block partition: P x P local windows (local attention).
    Grid partition:  P x P dilated groups (global attention at stride H/P).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        partition: str,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
    ):
        super().__init__()
        assert partition in ("block", "grid")
        assert dim % num_heads == 0
        self.partition = partition
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop_p = drop
        self.proj_drop = nn.Dropout(drop)

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def _attn(self, tokens: torch.Tensor) -> torch.Tensor:
        B, N, C = tokens.shape
        qkv = self.qkv(tokens).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop_p if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        P = self.window_size
        # Unconditional pad (zero pad is a no-op for 0,0) to stay compile-friendly:
        # a data-dependent `if pad_h or pad_w` breaks torch.compile graph tracing.
        pad_h = (P - H % P) % P
        pad_w = (P - W % P) % P
        x = F.pad(x, (0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w

        if self.partition == "block":
            tok = rearrange(x, "b c (h1 p1) (w1 p2) -> (b h1 w1) (p1 p2) c", p1=P, p2=P)
        else:
            tok = rearrange(x, "b c (h1 p1) (w1 p2) -> (b p1 p2) (h1 w1) c", p1=P, p2=P)

        tok = tok + self._attn(self.norm1(tok))
        tok = tok + self.mlp(self.norm2(tok))

        if self.partition == "block":
            y = rearrange(
                tok, "(b h1 w1) (p1 p2) c -> b c (h1 p1) (w1 p2)",
                b=B, h1=Hp // P, w1=Wp // P, p1=P, p2=P,
            )
        else:
            y = rearrange(
                tok, "(b p1 p2) (h1 w1) c -> b c (h1 p1) (w1 p2)",
                b=B, h1=Hp // P, w1=Wp // P, p1=P, p2=P,
            )
        return y[..., :H, :W]


class MaxViTBlock(nn.Module):
    """MBConv -> block-attention -> grid-attention."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        window_size: int,
        stride: int = 1,
        head_dim: int = 32,
        mbconv_expansion: float = 4.0,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
    ):
        super().__init__()
        self.mbconv = MBConv(in_dim, out_dim, stride=stride, expansion=mbconv_expansion)
        nh = _num_heads(out_dim, head_dim)
        self.block_attn = PartitionAttention(out_dim, nh, window_size, "block", mlp_ratio, drop)
        self.grid_attn = PartitionAttention(out_dim, nh, window_size, "grid", mlp_ratio, drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mbconv(x)
        x = self.block_attn(x)
        x = self.grid_attn(x)
        return x


class MaxViTStage(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        depth: int,
        window_size: int,
        downsample: bool = True,
        head_dim: int = 32,
        mbconv_expansion: float = 4.0,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        blocks: list[nn.Module] = []
        for i in range(depth):
            stride = 2 if (i == 0 and downsample) else 1
            d_in = in_dim if i == 0 else out_dim
            blocks.append(
                MaxViTBlock(
                    d_in, out_dim, window_size, stride=stride,
                    head_dim=head_dim, mbconv_expansion=mbconv_expansion,
                    mlp_ratio=mlp_ratio, drop=drop,
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return x


class ConvBlock(nn.Module):
    """Residual conv block used at full resolution where attention is wasteful."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        norm: str = "BATCH",
        act: str = "ReLU",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.shortcut = conv_norm_act(in_dim, out_dim, 1, norm, act)
        self.conv1 = conv_norm_act(in_dim, out_dim, 3, norm, act)
        self.conv2 = conv_norm_act(out_dim, out_dim, 3, norm, act)
        self.conv3 = conv_norm_act(out_dim, out_dim, 3, norm, act)
        self.conv4 = conv_norm_act(out_dim, out_dim, 3, norm, act)
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sc = self.shortcut(x)
        x = self.conv1(x)
        x = sc + self.dropout(self.conv2(x))
        x = x + self.dropout(self.conv4(self.conv3(x)))
        return x


class MaxViTDecoderStage(nn.Module):
    """Upsample (to skip size) -> concat skip -> 1x1 fuse -> MaxViT blocks."""

    def __init__(
        self,
        in_dim: int,
        skip_dim: int,
        out_dim: int,
        depth: int,
        window_size: int,
        head_dim: int = 32,
        mbconv_expansion: float = 4.0,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_dim + skip_dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
        )
        self.stage = MaxViTStage(
            out_dim, out_dim, depth=max(1, depth), window_size=window_size,
            downsample=False, head_dim=head_dim,
            mbconv_expansion=mbconv_expansion, mlp_ratio=mlp_ratio, drop=drop,
            gradient_checkpointing=gradient_checkpointing,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        return self.stage(x)


class ConvDecoderStage(nn.Module):
    """Full-resolution counterpart to MaxViTDecoderStage; no attention."""

    def __init__(
        self,
        in_dim: int,
        skip_dim: int,
        out_dim: int,
        norm: str = "BATCH",
        act: str = "ReLU",
        dropout: float = 0.0,
    ):
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


class MaxViTDecoder(nn.Module):
    """Mirror of the encoder: 2 MaxViT upsample stages + 1 conv upsample stage."""

    def __init__(
        self,
        layers: Sequence[int],
        out_channels: list[int],
        depths: Sequence[int],
        window_size: int,
        head_dim: int,
        mbconv_expansion: float,
        mlp_ratio: float,
        attn_drop: float,
        norm: str,
        act: str,
        dropout: float,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        d0, d1, d2, d3 = layers

        self.up2 = MaxViTDecoderStage(
            d3, d2, d2, depths[2], window_size,
            head_dim, mbconv_expansion, mlp_ratio, attn_drop,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.up1 = MaxViTDecoderStage(
            d2, d1, d1, depths[1], window_size,
            head_dim, mbconv_expansion, mlp_ratio, attn_drop,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.up0 = ConvDecoderStage(d1, d0, d0, norm=norm, act=act, dropout=dropout)

        final_norm = norm if (norm is not None and norm.lower() != "instance") else None
        self.final_blocks = nn.ModuleList(
            [conv_norm_act(d0, oc, 1, norm=final_norm, act=None) for oc in out_channels]
        )

    def forward(self, x: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        # skips ordered shallow -> deep: [s0 @ 1/1, s1 @ 1/2, s2 @ 1/4]
        x = self.up2(x, skips[2])
        x = self.up1(x, skips[1])
        x = self.up0(x, skips[0])
        return torch.cat([b(x) for b in self.final_blocks], dim=1)


class MaxViT(nn.Module):
    """Symmetric UNet-style encoder-decoder with MaxViT blocks.

    Encoder (4 levels, 3 downsamples):
        stage0  conv          in  -> d0 @ 1/1   (skip)
        stage1  MaxViT/s=2    d0  -> d1 @ 1/2   (skip)
        stage2  MaxViT/s=2    d1  -> d2 @ 1/4   (skip)
        stage3  MaxViT/s=2    d2  -> d3 @ 1/8   (bottleneck)

    Decoder mirrors that:
        up2  MaxViT  d3 -> d2 @ 1/4 (+skip d2)
        up1  MaxViT  d2 -> d1 @ 1/2 (+skip d1)
        up0  conv    d1 -> d0 @ 1/1 (+skip d0)

    `layers` is ordered shallow-to-deep (matches InstanSeg_UNet after its
    internal reversal). `out_channels` follows the same int / list /
    list-of-lists convention as InstanSeg_UNet.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels,
        layers: Sequence[int] = (32, 64, 128, 256),
        depths: Sequence[int] = (1, 2, 2, 2),
        window_size: int = 8,
        head_dim: int = 32,
        mbconv_expansion: float = 4.0,
        mlp_ratio: float = 4.0,
        norm: str = "BATCH",
        act: str = "ReLU",
        dropout: float = 0.0,
        attn_dropout: float | None = None,
        amp: bool = True,
        channels_last: bool = True,
        compile: bool = False,
        compile_mode: str = "default",
        amp_dtype: torch.dtype = torch.bfloat16,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        # Coerce to plain Python ints: callers may pass numpy arrays (see
        # model_loader), and np.int64 in nn.LayerNorm's normalized_shape
        # confuses torch.compile into treating it as a symbolic int.
        layers = [int(v) for v in layers]
        depths = [int(v) for v in depths]
        if len(layers) != 4:
            raise ValueError(f"MaxViT expects 4 feature levels, got {len(layers)}")
        if len(depths) != 4:
            raise ValueError(f"depths must have 4 entries, got {len(depths)}")
        d0, d1, d2, d3 = layers
        self.layers = layers
        self.window_size = int(window_size)
        # Default attn_dropout to the same value as dropout so a single
        # `dropout=` kwarg (or train.py's dropprob) applies uniformly.
        if attn_dropout is None:
            attn_dropout = dropout

        self.stage0 = ConvBlock(in_channels, d0, norm=norm, act=act, dropout=dropout)
        self.stage1 = MaxViTStage(
            d0, d1, depths[1], window_size, downsample=True,
            head_dim=head_dim, mbconv_expansion=mbconv_expansion,
            mlp_ratio=mlp_ratio, drop=attn_dropout,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.stage2 = MaxViTStage(
            d1, d2, depths[2], window_size, downsample=True,
            head_dim=head_dim, mbconv_expansion=mbconv_expansion,
            mlp_ratio=mlp_ratio, drop=attn_dropout,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.stage3 = MaxViTStage(
            d2, d3, depths[3], window_size, downsample=True,
            head_dim=head_dim, mbconv_expansion=mbconv_expansion,
            mlp_ratio=mlp_ratio, drop=attn_dropout,
            gradient_checkpointing=gradient_checkpointing,
        )

        if isinstance(out_channels, int):
            out_channels = [[out_channels]]
        if isinstance(out_channels[0], int):
            out_channels = [out_channels]

        self.decoders = nn.ModuleList(
            [
                MaxViTDecoder(
                    layers, oc, depths, window_size, head_dim,
                    mbconv_expansion, mlp_ratio, attn_dropout, norm, act, dropout,
                    gradient_checkpointing=gradient_checkpointing,
                )
                for oc in out_channels
            ]
        )

        self._amp = amp
        self._amp_dtype = amp_dtype
        self._channels_last = channels_last
        if channels_last:
            self.to(memory_format=torch.channels_last)
        # Compile the core forward, not the wrapper — autocast context must stay
        # outside the compiled region so dtype promotion happens per-call.
        self._core = torch.compile(self._run, mode=compile_mode) if compile else self._run

    def get_embedding_tap(self) -> nn.Module:
        return self.stage3

    def _run(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.stage0(x)
        s1 = self.stage1(s0)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        skips = [s0, s1, s2]
        return torch.cat([dec(s3, skips) for dec in self.decoders], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        if self._amp and x.is_cuda:
            in_dtype = x.dtype
            with torch.autocast("cuda", dtype=self._amp_dtype):
                y = self._core(x)
            # Cast back so downstream code (viz, .numpy(), fp32 losses) sees
            # the input dtype; compute stays in low precision inside autocast.
            return y.to(in_dtype)
        return self._core(x)


# Preset sizes. Targets (with the default symmetric UNet decoder):
#   tiny   ~8M    — layers (32,64,128,256),    depths (1,2,2,2)
#   base   ~47M   — layers (64,128,256,512),   depths (2,2,5,2)
#   large  ~309M  — layers (128,256,512,1024), depths (2,5,10,2)
# Large enables gradient_checkpointing by default — a 1024-wide bottleneck
# with mbconv_expansion=4 explodes activation memory on 256x256 inputs
# otherwise. Override by passing gradient_checkpointing=False.
_MAXVIT_PRESETS: dict[str, dict] = {
    "tiny":  dict(layers=(32,  64,  128,  256),  depths=(1, 2,  2, 2), gradient_checkpointing=False),
    "base":  dict(layers=(64,  128, 256,  512),  depths=(2, 2,  5, 2), gradient_checkpointing=False),
    "large": dict(layers=(128, 256, 512, 1024),  depths=(2, 5, 10, 2), gradient_checkpointing=True),
}


def _make_preset(name: str, in_channels: int, out_channels, **overrides) -> MaxViT:
    cfg = dict(_MAXVIT_PRESETS[name])
    cfg.update(overrides)
    return MaxViT(in_channels=in_channels, out_channels=out_channels, **cfg)


def maxvit_tiny(in_channels: int, out_channels, **kwargs) -> MaxViT:
    return _make_preset("tiny", in_channels, out_channels, **kwargs)


def maxvit_base(in_channels: int, out_channels, **kwargs) -> MaxViT:
    return _make_preset("base", in_channels, out_channels, **kwargs)


def maxvit_large(in_channels: int, out_channels, **kwargs) -> MaxViT:
    return _make_preset("large", in_channels, out_channels, **kwargs)


if __name__ == "__main__":
    model = MaxViT(
        in_channels=3,
        out_channels=[[2, 2, 1]],
        layers=(32, 64, 128, 256),
        depths=(1, 2, 2, 2),
        window_size=8,
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"MaxViT parameters: {n_params:.2f}M")
    with torch.no_grad():
        for H, W in [(256, 256), (128, 128), (192, 160)]:
            y = model(torch.randn(1, 3, H, W))
            assert y.shape[-2:] == (H, W), f"Bad output shape: {y.shape} for input ({H},{W})"
            print(f"Input (1,3,{H},{W}) -> Output {tuple(y.shape)}")

    if torch.cuda.is_available():
        fast = MaxViT(
            in_channels=3,
            out_channels=[[2, 2, 1]],
            layers=(32, 64, 128, 256),
            depths=(1, 2, 2, 2),
            amp=True,
            channels_last=True,
            compile=True,
        ).cuda().eval()
        with torch.no_grad():
            y = fast(torch.randn(1, 3, 256, 256, device="cuda"))
        print(f"Fast-path (amp+channels_last+compile) -> {tuple(y.shape)}")
