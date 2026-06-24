"""Masked-image-modeling (MAE-style) pretraining for InstanSeg backbones.

Surgical add-on: reuses the existing data pipeline, augmentations, LR
schedule, DDP and batch-size-finder unchanged. Only the objective changes.

Because the InstanSeg backbones (e.g. MaxViT) are conv-attention U-Nets rather
than plain ViTs, we cannot drop the masked tokens from the encoder (the conv
stem needs a dense grid). So this is *masked image modeling*: random patches of
the (already normalized) input are zeroed, the full encoder-decoder reconstructs
the image, and an MSE loss is applied to the masked patches only. Same
self-supervised objective and pretraining benefit as MAE, just without the
token-dropping speedup.

Wiring (see ``scripts/train.py --mae``):
    model   = MAEWrapper(backbone)              # backbone outputs dim_in channels
    loss_fn = partial(mae_loss_fn, patch_size=..., norm_target=...)
The train/eval step is unchanged: ``output = model(x); loss = loss_fn(output, labels).mean()``.
``labels`` (the segmentation maps) are ignored by the loss.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def _patchify(imgs: torch.Tensor, patch_size: int) -> torch.Tensor:
    """``(B, C, H, W)`` -> ``(B, n_patches, C*p*p)`` (row-major over the patch grid)."""
    b, c, h, w = imgs.shape
    p = patch_size
    gh, gw = h // p, w // p
    x = imgs.reshape(b, c, gh, p, gw, p)
    x = x.permute(0, 2, 4, 1, 3, 5).reshape(b, gh * gw, c * p * p)
    return x


def _unpatchify(patches: torch.Tensor, channels: int, h: int, w: int, patch_size: int) -> torch.Tensor:
    """Inverse of :func:`_patchify`: ``(B, n_patches, C*p*p)`` -> ``(B, C, H, W)``."""
    b = patches.shape[0]
    p = patch_size
    gh, gw = h // p, w // p
    x = patches.reshape(b, gh, gw, channels, p, p)
    x = x.permute(0, 3, 1, 4, 2, 5).reshape(b, channels, h, w)
    return x


@torch.no_grad()
def mae_make_panels(
    output: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    patch_size: int = 16,
    norm_target: bool = True,
    index: int = 0,
):
    """Build diagnostic panels for one sample of an MAE batch.

    Returns ``(images, titles)`` for :func:`instanseg.utils.visualization.show_images`:
    original, masked input (visible patches only), reconstruction, and the
    canonical MAE composite (visible original patches + predicted masked patches).
    When ``norm_target`` is set the prediction is de-normalized back into pixel
    space using each target patch's own mean/std so it is directly comparable.
    """
    recon, target, mask_pix = output
    _, c, h, w = target.shape
    p = patch_size

    pred_p = _patchify(recon, p)
    if norm_target:
        tgt_p = _patchify(target, p)
        mean = tgt_p.mean(dim=-1, keepdim=True)
        std = (tgt_p.var(dim=-1, keepdim=True) + 1e-6).sqrt()
        pred_p = pred_p * std + mean
    pred_img = _unpatchify(pred_p, c, h, w, p)

    masked_input = target * (1.0 - mask_pix)
    composite = target * (1.0 - mask_pix) + pred_img * mask_pix

    images = [target[index], masked_input[index], pred_img[index], composite[index]]
    titles = ["Original", "Masked input", "Reconstruction", "Recon + visible"]
    return images, titles


def _random_patch_mask(
    b: int,
    h: int,
    w: int,
    patch_size: int,
    mask_ratio: float,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Per-sample random patch mask upsampled to pixels. ``1`` = masked, ``0`` = visible.

    Returns ``(B, 1, H, W)``. Exactly ``round(mask_ratio * n_patches)`` patches
    are masked per sample (random subset).
    """
    p = patch_size
    gh, gw = h // p, w // p
    n = gh * gw
    n_mask = int(round(mask_ratio * n))
    noise = torch.rand(b, n, device=device, generator=generator)
    ids = noise.argsort(dim=1)  # ascending; first n_mask are masked
    mask = torch.zeros(b, n, device=device)
    mask.scatter_(1, ids[:, :n_mask], 1.0)
    mask = mask.view(b, 1, gh, gw)
    return F.interpolate(mask, size=(h, w), mode="nearest")


class MAEWrapper(nn.Module):
    """Wrap a dense backbone (image -> image) into a masked-reconstruction model.

    ``forward(x)`` zeroes a random subset of input patches, runs the backbone,
    and returns ``(reconstruction, target, mask)`` for :func:`mae_loss_fn`.
    Parameter-free (masked pixels are zeroed), so it composes cleanly with
    ``static_graph=True`` DDP and adds nothing to the optimizer.
    """

    def __init__(self, backbone: nn.Module, patch_size: int = 16, mask_ratio: float = 0.6):
        super().__init__()
        self.backbone = backbone
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        # Carry through metadata some downstream code reads off the model.
        self.pixel_size = getattr(backbone, "pixel_size", None)

    def forward(
        self, x: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        p = self.patch_size
        if h % p != 0 or w % p != 0:
            raise ValueError(
                f"MAE patch_size={p} must divide the crop size, got H={h}, W={w}. "
                f"Set --window_size / --mae_patch_size accordingly."
            )
        mask = _random_patch_mask(b, h, w, p, self.mask_ratio, x.device)
        recon = self.backbone(x * (1.0 - mask))
        return recon, x, mask


def mae_loss_fn(
    output: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    labels: Optional[torch.Tensor] = None,
    *,
    patch_size: int = 16,
    norm_target: bool = True,
) -> torch.Tensor:
    """MSE reconstruction loss over masked patches. ``labels`` is ignored.

    Returns a scalar tensor (``.mean()`` in the train loop is then a no-op). When
    ``norm_target`` is set, each target patch is normalized to zero-mean/unit-var
    over its own pixels before the loss (the MAE-paper default; the model then
    predicts in normalized-patch space).
    """
    recon, target, mask_pix = output
    p = patch_size

    pred_p = _patchify(recon, p)        # (B, n, C*p*p)
    tgt_p = _patchify(target, p)
    if norm_target:
        mean = tgt_p.mean(dim=-1, keepdim=True)
        var = tgt_p.var(dim=-1, keepdim=True)
        tgt_p = (tgt_p - mean) / (var + 1e-6).sqrt()

    loss = (pred_p - tgt_p) ** 2
    loss = loss.mean(dim=-1)            # (B, n) per-patch mean over pixels/channels

    mask_p = _patchify(mask_pix, p).mean(dim=-1)  # (B, n) in {0, 1}
    denom = mask_p.sum().clamp(min=1.0)
    return (loss * mask_p).sum() / denom
