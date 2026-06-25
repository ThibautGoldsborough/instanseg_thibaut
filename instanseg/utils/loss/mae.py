"""Masked-image-modeling pretraining + MAE->seg transfer. See CLAUDE.md `--mae` note."""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def _patchify(imgs: torch.Tensor, p: int) -> torch.Tensor:
    """(B,C,H,W) -> (B, n_patches, C*p*p)."""
    b, c, h, w = imgs.shape
    gh, gw = h // p, w // p
    x = imgs.reshape(b, c, gh, p, gw, p)
    return x.permute(0, 2, 4, 1, 3, 5).reshape(b, gh * gw, c * p * p)


def _unpatchify(patches: torch.Tensor, c: int, h: int, w: int, p: int) -> torch.Tensor:
    """Inverse of _patchify."""
    b = patches.shape[0]
    gh, gw = h // p, w // p
    x = patches.reshape(b, gh, gw, c, p, p)
    return x.permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w)


def _random_patch_mask(b: int, h: int, w: int, p: int, mask_ratio: float,
                       device: torch.device) -> torch.Tensor:
    """Per-sample pixel mask (B,1,H,W), 1=masked. Masks round(mask_ratio*n_patches) patches."""
    gh, gw = h // p, w // p
    n = gh * gw
    n_mask = int(round(mask_ratio * n))
    ids = torch.rand(b, n, device=device).argsort(dim=1)  # first n_mask -> masked
    mask = torch.zeros(b, n, device=device)
    mask.scatter_(1, ids[:, :n_mask], 1.0)
    return F.interpolate(mask.view(b, 1, gh, gw), size=(h, w), mode="nearest")


class MAEWrapper(nn.Module):
    """Wrap a dense image->image backbone into a masked-reconstruction model.

    forward(x) zeroes a random subset of input patches, runs the backbone, and
    returns (reconstruction, target, mask) for mae_loss_fn. Parameter-free.
    """

    def __init__(self, backbone: nn.Module, patch_size: int = 16, mask_ratio: float = 0.6):
        super().__init__()
        self.backbone = backbone
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.pixel_size = getattr(backbone, "pixel_size", None)  # carry metadata through

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        p = self.patch_size
        if h % p != 0 or w % p != 0:
            raise ValueError(f"MAE patch_size={p} must divide the crop size, got H={h}, W={w}.")
        mask = _random_patch_mask(b, h, w, p, self.mask_ratio, x.device)
        return self.backbone(x * (1.0 - mask)), x, mask


def mae_loss_fn(output: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                labels: Optional[torch.Tensor] = None, *,
                patch_size: int = 16, norm_target: bool = True) -> torch.Tensor:
    """MSE over masked patches only (labels ignored). norm_target: per-patch standardize."""
    recon, target, mask_pix = output
    p = patch_size
    pred_p, tgt_p = _patchify(recon, p), _patchify(target, p)
    if norm_target:
        mean = tgt_p.mean(dim=-1, keepdim=True)
        var = tgt_p.var(dim=-1, keepdim=True)
        tgt_p = (tgt_p - mean) / (var + 1e-6).sqrt()
    loss = ((pred_p - tgt_p) ** 2).mean(dim=-1)        # (B, n) per-patch
    mask_p = _patchify(mask_pix, p).mean(dim=-1)        # (B, n) in {0,1}
    return (loss * mask_p).sum() / mask_p.sum().clamp(min=1.0)


@torch.no_grad()
def mae_make_panels(output: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                    patch_size: int = 16, norm_target: bool = True, index: int = 0):
    """Diagnostic panels (images, titles) for show_images: original / masked / recon / composite.

    Prediction is de-normalized with each target patch's mean/std when norm_target.
    """
    recon, target, mask_pix = output
    _, c, h, w = target.shape
    p = patch_size
    pred_p = _patchify(recon, p)
    if norm_target:
        tgt_p = _patchify(target, p)
        pred_p = pred_p * (tgt_p.var(dim=-1, keepdim=True) + 1e-6).sqrt() + tgt_p.mean(dim=-1, keepdim=True)
    pred_img = _unpatchify(pred_p, c, h, w, p)
    masked_input = target * (1.0 - mask_pix)
    composite = masked_input + pred_img * mask_pix
    images = [target[index], masked_input[index], pred_img[index], composite[index]]
    return images, ["Original", "Masked input", "Reconstruction", "Recon + visible"]


def load_mae_backbone(model: nn.Module, ckpt_path, verbose: bool = True) -> nn.Module:
    """Transfer matching (name+shape) backbone tensors from an MAE checkpoint, strict=False.

    Strips backbone./module. prefixes; recon heads (shape-mismatch) and pixel_classifier
    (absent) stay at init. ckpt_path may be the run folder or the model_weights.pth file.
    """
    from pathlib import Path
    p = Path(ckpt_path)
    if p.is_dir():
        p = p / "model_weights.pth"
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    def _strip(k: str) -> str:
        for pre in ("module.", "backbone."):
            if k.startswith(pre):
                k = k[len(pre):]
        return k

    src = {_strip(k): v for k, v in sd.items()}
    tgt = model.state_dict()
    matched = {k: src[k] for k, v in tgt.items()
               if k in src and tuple(src[k].shape) == tuple(v.shape)}
    model.load_state_dict(matched, strict=False)
    if verbose:
        print(f"[mae_init] transferred {len(matched)}/{len(tgt)} tensors from {p} "
              f"({len(tgt) - len(matched)} kept random: recon heads + pixel_classifier)")
    return model
