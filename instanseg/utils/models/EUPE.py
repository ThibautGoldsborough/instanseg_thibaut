from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from instanseg.utils.models.InstanSeg_UNet import Decoder, EncoderBlock


# GitHub repo spec for torch.hub.load (source='github'). Torch auto-checks out to
# ~/.cache/torch/hub/, so no manual clone is required. Override with env var if
# you need a fork/branch (e.g. "myorg/eupe:main").
EUPE_GITHUB_REPO = os.environ.get("EUPE_GITHUB_REPO", "facebookresearch/eupe")

EUPE_CONVNEXT_MODELS = {
    "eupe_convnext_tiny",
    "eupe_convnext_small",
    "eupe_convnext_base",
}

# Default pretrained checkpoints hosted on HuggingFace (the URL inside the EUPE
# repo points at fbaipublicfiles which currently returns HTTP 403).
EUPE_CONVNEXT_DEFAULT_WEIGHTS = {
    "eupe_convnext_tiny":  "https://huggingface.co/facebook/EUPE-ConvNeXt-T/resolve/main/EUPE-ConvNeXt-T.pt",
    "eupe_convnext_small": "https://huggingface.co/facebook/EUPE-ConvNeXt-S/resolve/main/EUPE-ConvNeXt-S.pt",
    "eupe_convnext_base":  "https://huggingface.co/facebook/EUPE-ConvNeXt-B/resolve/main/EUPE-ConvNeXt-B.pt",
}


def _load_eupe_convnext(
    eupe_model: str,
    weights: Optional[str] = None,
    pretrained: bool = True,
    github_repo: str = EUPE_GITHUB_REPO,
) -> nn.Module:
    """Load an EUPE ConvNeXt backbone via ``torch.hub.load`` from GitHub.

    Torch caches the checkout under ``~/.cache/torch/hub/``; no manual clone
    required. By default the pretrained LVD1689M weights are pulled from
    HuggingFace. Pass ``weights`` to use a specific checkpoint path/URL, or
    ``pretrained=False`` to skip weight loading entirely.
    """
    if eupe_model not in EUPE_CONVNEXT_MODELS:
        raise ValueError(
            f"Unknown EUPE ConvNeXt model '{eupe_model}'. "
            f"Must be one of {sorted(EUPE_CONVNEXT_MODELS)}"
        )

    kwargs = {"pretrained": pretrained}
    if pretrained:
        # Prefer an explicit override; otherwise fall back to the HF default URL,
        # since EUPE's built-in fbaipublicfiles URL currently 403s.
        resolved_weights = weights if weights is not None else EUPE_CONVNEXT_DEFAULT_WEIGHTS[eupe_model]
        kwargs["weights"] = resolved_weights
    return torch.hub.load(github_repo, eupe_model, source="github", **kwargs)


class EUPE(nn.Module):
    """InstanSeg head on a pretrained EUPE ConvNeXt backbone.

    Architecture (256x256 input):
        Block1 (EncoderBlock, no pool):  in_channels -> 16ch @ 1/1
        ConvNeXt stem + stage0:           16ch       -> dims[0] @ 1/4
        ConvNeXt stage1:                              -> dims[1] @ 1/8
        ConvNeXt stage2:                              -> dims[2] @ 1/16
        ConvNeXt stage3:                              -> dims[3] @ 1/32
        Neck (1x1 -> 3x3):                dims[3]   -> 256ch @ 1/32
        Decoder (5 upsamples, x2 each):   256ch     -> ... -> 16ch -> dim_out @ 1/1

    Skip connections feed the decoder in order of decreasing resolution:
        [block1 @ 1/1, block1-pool @ 1/2, stage0 @ 1/4, stage1 @ 1/8, stage2 @ 1/16]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels,
        eupe_model: str,
        layers=(64,),  # unused, kept for API compatibility with other InstanSeg models
        norm: str = "BATCH",
        dropout: float = 0,
        act: str = "ReLU",
        eupe_weights: Optional[str] = None,
        eupe_pretrained: bool = True,
        eupe_github_repo: str = EUPE_GITHUB_REPO,
    ):
        super().__init__()

        self.eupe_model = eupe_model
        block1_ch = 16
        neck_ch = 256
        self.block1_ch = block1_ch
        self.neck_ch = neck_ch

        self.block1 = EncoderBlock(
            in_channels, block1_ch, pool=False, norm=norm, act=act, dropout=dropout
        )

        self.backbone = _load_eupe_convnext(
            eupe_model,
            weights=eupe_weights,
            pretrained=eupe_pretrained,
            github_repo=eupe_github_repo,
        )
        # Backbone is trainable by default; train.py's --freeze_main_model flag
        # (and callbacks around warmup) manage freezing via freeze_backbone().

        dims = self.backbone.embed_dims  # [C0, C1, C2, C3], strides [4, 8, 16, 32]

        self._modify_convnext_stem(block1_ch)

        self.neck = nn.Sequential(
            nn.Conv2d(dims[-1], neck_ch, 1, bias=False),
            nn.BatchNorm2d(neck_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(neck_ch, neck_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(neck_ch),
            nn.ReLU(inplace=True),
        )

        # Decoder channel plan matches ConvNeXt per-stage dims so skip channels align.
        stride2_ch = max(block1_ch * 2, dims[0] // 2)
        dec_layers = [neck_ch, dims[2], dims[1], dims[0], stride2_ch, block1_ch]

        # Skip projections: decoder blocks consume skips[::-1], so indices line up as:
        #   block 0 (out=dec_layers[1]=dims[2]) <- skips[4]  = proj(stage2, dims[2]->dec_layers[1])
        #   block 1 (out=dec_layers[2]=dims[1]) <- skips[3]  = proj(stage1, dims[1]->dec_layers[2])
        #   block 2 (out=dec_layers[3]=dims[0]) <- skips[2]  = proj(stage0, dims[0]->dec_layers[3])
        #   block 3 (out=dec_layers[4]=stride2) <- skips[1]  = proj(block1-pool, block1_ch->stride2)
        #   block 4 (out=dec_layers[5]=block1)  <- skips[0]  = block1 directly (no projection)
        self.stage2_proj = nn.Conv2d(dims[2], dec_layers[1], 1)
        self.stage1_proj = nn.Conv2d(dims[1], dec_layers[2], 1)
        self.stage0_proj = nn.Conv2d(dims[0], dec_layers[3], 1)
        self.stride2_proj = nn.Conv2d(block1_ch, dec_layers[4], 1)

        if type(out_channels) == int:
            out_channels = [[out_channels]]
        if type(out_channels[0]) == int:
            out_channels = [out_channels]

        self.decoders = nn.ModuleList([
            Decoder(dec_layers, out_ch, norm, act, dropout=dropout)
            for out_ch in out_channels
        ])

        print(
            f"EUPE[{self.eupe_model}]: dims={dims}, dec_layers={dec_layers}"
        )

    def _modify_convnext_stem(self, in_chans: int):
        """Adapt the ConvNeXt stem's first conv to accept `in_chans` inputs."""
        stem = self.backbone.downsample_layers[0]
        old_conv: nn.Conv2d = stem[0]
        old_w = old_conv.weight.data  # (C0, 3, 4, 4)
        if in_chans == old_w.shape[1]:
            return
        new_w = old_w.mean(dim=1, keepdim=True).repeat(1, in_chans, 1, 1)
        new_conv = nn.Conv2d(
            in_chans,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
        )
        new_conv.weight = nn.Parameter(new_w)
        if old_conv.bias is not None:
            new_conv.bias = nn.Parameter(old_conv.bias.data.clone())
        stem[0] = new_conv

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def _run_backbone(self, x: torch.Tensor):
        stage_feats = []
        for i in range(4):
            x = self.backbone.downsample_layers[i](x)
            x = self.backbone.stages[i](x)
            stage_feats.append(x)
        neck_out = self.neck(stage_feats[-1])
        return stage_feats, neck_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        block1_out = self.block1(x)  # (B, 16, H, W)
        stage_feats, neck_out = self._run_backbone(block1_out)

        stride2_skip = self.stride2_proj(
            F.avg_pool2d(block1_out, kernel_size=2, stride=2)
        )
        skips = [
            block1_out,                              # 1/1
            stride2_skip,                            # 1/2
            self.stage0_proj(stage_feats[0]),        # 1/4
            self.stage1_proj(stage_feats[1]),        # 1/8
            self.stage2_proj(stage_feats[2]),        # 1/16
        ]
        return torch.cat([dec(neck_out, skips) for dec in self.decoders], dim=1)


if __name__ == "__main__":
    for model_name in sorted(EUPE_CONVNEXT_MODELS):
        print(f"\n--- {model_name} ---")
        model = EUPE(
            in_channels=3,
            out_channels=[[2, 2, 1]],
            eupe_model=model_name,
            eupe_pretrained=False,
        )
        model.eval()
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"Parameters: {n_params:.1f}M")
        with torch.no_grad():
            for H, W in [(256, 256), (128, 128)]:
                y = model(torch.randn(1, 3, H, W))
                assert y.shape[-2:] == (H, W), f"Bad output shape: {y.shape}"
                print(f"Input ({H},{W}) -> Output {tuple(y.shape)} OK")
