"""Equivariance consistency loss for offset-prediction models.

Self-contained module. Single public function `consistency_loss(model, images)`
returns a scalar loss enforcing that the model's per-pixel (y, x) offset output
is equivariant under a random composite augmentation T (affine homography
followed by elastic deformation), sampled independently for each batch element.

Coordinate convention: matches InstanSeg's `generate_coordinate_map`. The
internal position grid uses `linspace(0, H*coord_per_pixel, H)` (default
`coord_per_pixel = 64/256`), so 1 coord unit ≈ 4 pixels. Model output is
interpreted as a displacement in those same units. Augmentation magnitudes
(elastic_magnitude, max_translation_frac×extent, perspective_jitter_frac×extent)
are likewise in coord units; the smoothing kernel sigma is in pixels because
it acts on the H×W tensor grid.

The forward path:
    offset_orig = model(images)
    image_aug   = T(images)              # affine then elastic, per-element
    offset_aug  = model(image_aug)
    offset_aug_inverted = T_inv(offset_aug)
    cons = MSE(offset_aug_inverted, offset_orig) on pixels where T(p) is in-bounds

The inverse for elastic uses a first-order approximation T_inv(p) ≈ p − δ(p),
which is exact in the limit of small δ. The inverse for the homography is
exact. Boundary pixels (where T(p) lands off-canvas) are masked out via the
returned valid_mask, so they don't contribute fake error to the loss.

Model contract:
    model(x: torch.Tensor[(B, C_in, H, W)]) → torch.Tensor[(B, 2, H, W)]
    where output channel 0 = y-component of the offset (in coord units),
    channel 1 = x-component. For an InstanSeg backbone, that means
    `(sigmoid(raw[:, :2]) - 0.5) * 8` (the InstanSeg displacement formula).
    Passing raw logits is mathematically self-consistent but not physically
    meaningful — the loss won't capture geometric equivariance accurately.

Usage:
    from consistency_loss import consistency_loss
    loss = consistency_loss(model, images)
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_homography_yx(src_yx: torch.Tensor, dst_yx: torch.Tensor) -> torch.Tensor:
    """DLT homography solver in (y, x) order. src/dst: (4, 2). Returns 3×3."""
    rows: list[list[float]] = []
    bvec: list[float] = []
    for (ys, xs), (yd, xd) in zip(src_yx.tolist(), dst_yx.tolist()):
        rows.append([ys, xs, 1, 0, 0, 0, -yd * ys, -yd * xs])
        bvec.append(yd)
        rows.append([0, 0, 0, ys, xs, 1, -xd * ys, -xd * xs])
        bvec.append(xd)
    A = torch.tensor(rows, dtype=torch.float32)
    b = torch.tensor(bvec, dtype=torch.float32)
    h = torch.linalg.solve(A, b)
    return torch.cat([h, torch.tensor([1.0])]).reshape(3, 3)


def _sample_homography(
    rng: torch.Generator, H_extent: float, W_extent: float, *,
    axis_aligned_rotation: bool,
    scale_range: tuple[float, float],
    max_translation_frac: float,
    p_flip: float,
    perspective_jitter_frac: float,
    rotation_intensity: float = 1.0,
) -> torch.Tensor:
    """Sample a 3×3 composite affine + perspective homography in (y, x) coords.

    H_extent / W_extent are the canvas size in whatever coordinate units the
    caller is operating in (e.g. pixels or InstanSeg's 64-per-256-pixel units).
    Translation, jitter, and corner positions are all expressed in those units.

    rotation_intensity scales rotation magnitude. For axis-aligned mode it sets
    the probability of applying a non-identity 90° rotation (so at intensity=1
    P(k>0)=0.75, matching uniform-{0,1,2,3} sampling; at intensity=0 always k=0).
    For continuous mode it scales the angle range to [-π·intensity, π·intensity].
    """
    cy = H_extent / 2.0
    cx = W_extent / 2.0
    src = torch.tensor(
        [[0, 0], [0, W_extent], [H_extent, 0], [H_extent, W_extent]], dtype=torch.float32
    )

    def u(scale: float = 1.0) -> float:
        return float((torch.rand(1, generator=rng).item() * 2 - 1) * scale)

    if axis_aligned_rotation:
        if torch.rand(1, generator=rng).item() < 0.75 * rotation_intensity:
            k = int(torch.randint(1, 4, (1,), generator=rng).item())
        else:
            k = 0
        theta = k * math.pi / 2
    else:
        theta = u(math.pi * rotation_intensity)
    c_t, s_t = math.cos(theta), math.sin(theta)
    scale_y = scale_range[0] + (scale_range[1] - scale_range[0]) * float(
        torch.rand(1, generator=rng).item()
    )
    scale_x = scale_range[0] + (scale_range[1] - scale_range[0]) * float(
        torch.rand(1, generator=rng).item()
    )
    sy = -1.0 if torch.rand(1, generator=rng).item() < p_flip else 1.0
    sx = -1.0 if torch.rand(1, generator=rng).item() < p_flip else 1.0
    dy = u(max_translation_frac * H_extent)
    dx = u(max_translation_frac * W_extent)

    centered = src - torch.tensor([cy, cx])
    rot_y = c_t * centered[:, 0] + s_t * centered[:, 1]
    rot_x = -s_t * centered[:, 0] + c_t * centered[:, 1]
    rotated = torch.stack([rot_y, rot_x], dim=-1) * torch.tensor([scale_y, scale_x])
    rotated = rotated * torch.tensor([sy, sx])
    dst = rotated + torch.tensor([cy + dy, cx + dx])

    pjitter = (torch.rand(4, 2, generator=rng) * 2 - 1) * perspective_jitter_frac * H_extent
    dst = dst + pjitter
    return _compute_homography_yx(src, dst)


def _sample_elastic_field_batch(
    rng: torch.Generator, B: int, H: int, W: int,
    magnitude: float, smooth_sigma: float, device: torch.device,
) -> torch.Tensor:
    """Smooth random displacement field, batched. Returns (B, 2, H, W).
    Uses separable 1D Gaussians (much faster than full 2D conv for large σ)."""
    raw = (torch.randn(B, 2, H, W, generator=rng) * magnitude).to(device)
    radius = max(1, int(np.ceil(3.0 * smooth_sigma)))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    g = torch.exp(-(coords ** 2) / (2.0 * smooth_sigma * smooth_sigma))
    g = g / g.sum()
    k_v = g.reshape(1, 1, 2 * radius + 1, 1)
    k_h = g.reshape(1, 1, 1, 2 * radius + 1)
    x = raw.reshape(B * 2, 1, H, W)
    x = F.conv2d(x, k_v, padding=(radius, 0))
    x = F.conv2d(x, k_h, padding=(0, radius))
    return x.reshape(B, 2, H, W).float()


# ---------------------------------------------------------------------------
# Aug forward + offset inverse
# ---------------------------------------------------------------------------

def _apply_aug_and_make_unwarp(
    images: torch.Tensor,
    coord_grid: torch.Tensor,
    H_extent: float,
    W_extent: float,
    rng: torch.Generator,
    *,
    axis_aligned_rotation: bool,
    scale_range: tuple[float, float],
    max_translation_frac: float,
    p_flip: float,
    perspective_jitter_frac: float,
    elastic_magnitude: float,
    elastic_smooth_sigma: float,
    rotation_intensity: float = 1.0,
):
    """Sample T per batch element (affine then elastic). Apply T to image.

    All geometry (homography, elastic field, position grid) is in InstanSeg
    coordinate units, where pixel idx i maps to coord i*64/256. H_extent and
    W_extent give the canvas size in those units (= H*64/256, W*64/256).

    Returns (image_aug, unwarp_fn) where unwarp_fn(offset_aug) -> (offset_in_orig_frame, valid_mask).
    """
    B, _, H, W = images.shape
    device = images.device

    # Per-element homographies (in coord units)
    H_mats_list = [
        _sample_homography(
            rng, H_extent, W_extent,
            axis_aligned_rotation=axis_aligned_rotation,
            scale_range=scale_range,
            max_translation_frac=max_translation_frac,
            p_flip=p_flip,
            perspective_jitter_frac=perspective_jitter_frac,
            rotation_intensity=rotation_intensity,
        )
        for _ in range(B)
    ]
    H_mats = torch.stack(H_mats_list, dim=0).to(device).float()
    H_invs = torch.linalg.inv(H_mats)

    # Elastic field stored at pixel grid; values are displacements in coord units.
    # smooth_sigma is in pixels (the smoothing acts on the H×W tensor grid).
    deltas = _sample_elastic_field_batch(
        rng, B, H, W, elastic_magnitude, elastic_smooth_sigma, device
    )

    # T_full(p_orig) = H @ p_orig + δ(H @ p_orig), all in coord units.
    yx = coord_grid.reshape(2, -1)
    homog = torch.cat([yx, torch.ones(1, yx.shape[1], device=device)], dim=0).float()
    homog_b = homog.unsqueeze(0).expand(B, -1, -1)
    p_aff_homog = H_mats @ homog_b
    p_aff_flat = p_aff_homog[:, :2] / p_aff_homog[:, 2:3]
    p_affine = p_aff_flat.reshape(B, 2, H, W)

    # grid_sample normalization: with align_corners=True and a coord-unit input,
    # value 0 → -1, value H_extent → +1 (last pixel sits at coord H_extent).
    p_aff_y_norm = 2.0 * p_affine[:, 0] / H_extent - 1.0
    p_aff_x_norm = 2.0 * p_affine[:, 1] / W_extent - 1.0
    grid_aff = torch.stack([p_aff_x_norm, p_aff_y_norm], dim=-1)
    delta_at_p_aff = F.grid_sample(
        deltas, grid_aff, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    p_aug = p_affine + delta_at_p_aff  # (B, 2, H, W) in coord units

    valid_mask = (
        (p_aug[:, 0] >= 0) & (p_aug[:, 0] <= H_extent)
        & (p_aug[:, 1] >= 0) & (p_aug[:, 1] <= W_extent)
    )

    # Forward image warp: image_aug[p] = image[T_full_inv(p)]; reflection padding
    # so off-canvas pixels are filled with mirrored real content (no black borders).
    pos_grid_b = coord_grid.unsqueeze(0).expand(B, -1, -1, -1)
    q1 = pos_grid_b - deltas  # T_elastic_inv (first-order)
    q1_flat = q1.reshape(B, 2, -1).float()
    ones = torch.ones(B, 1, q1_flat.shape[2], device=device, dtype=q1_flat.dtype)
    homog_q1 = torch.cat([q1_flat, ones], dim=1)
    q2_homog = H_invs @ homog_q1
    q2_flat = q2_homog[:, :2] / q2_homog[:, 2:3]
    src_pos = q2_flat.reshape(B, 2, H, W)
    src_y_norm = 2.0 * src_pos[:, 0] / H_extent - 1.0
    src_x_norm = 2.0 * src_pos[:, 1] / W_extent - 1.0
    src_grid = torch.stack([src_x_norm, src_y_norm], dim=-1)
    image_aug = F.grid_sample(
        images.float(), src_grid, mode="bilinear",
        padding_mode="reflection", align_corners=True,
    )

    def unwarp(offset_aug: torch.Tensor):
        Bo, _, Hh, Ww = offset_aug.shape
        offset_aug_f = offset_aug.float()
        p_aug_y_norm_local = 2.0 * p_aug[:, 0] / H_extent - 1.0
        p_aug_x_norm_local = 2.0 * p_aug[:, 1] / W_extent - 1.0
        grid_aug = torch.stack([p_aug_x_norm_local, p_aug_y_norm_local], dim=-1)
        sampled_offset = F.grid_sample(
            offset_aug_f, grid_aug, mode="bilinear",
            padding_mode="zeros", align_corners=True,
        )
        target_aug = p_aug + sampled_offset

        ta_y_norm = 2.0 * target_aug[:, 0] / H_extent - 1.0
        ta_x_norm = 2.0 * target_aug[:, 1] / W_extent - 1.0
        grid_target = torch.stack([ta_x_norm, ta_y_norm], dim=-1)
        delta_at_target = F.grid_sample(
            deltas, grid_target, mode="bilinear",
            padding_mode="zeros", align_corners=True,
        )
        q = target_aug - delta_at_target

        flat = q.reshape(Bo, 2, -1)
        ones_b = torch.ones(Bo, 1, flat.shape[2], device=flat.device, dtype=flat.dtype)
        homog_q = torch.cat([flat, ones_b], dim=1)
        target_orig_homog = H_invs @ homog_q
        target_orig_flat = target_orig_homog[:, :2] / target_orig_homog[:, 2:3]
        target_orig = target_orig_flat.reshape(Bo, 2, Hh, Ww)
        return target_orig - pos_grid_b, valid_mask

    return image_aug, unwarp


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def consistency_loss(
    model,
    images: torch.Tensor,
    *,
    rng: torch.Generator | None = None,
    transformation_intensity: float = 0.1,
    axis_aligned_rotation: bool = True,
    scale_range: tuple[float, float] = (0.7, 1.4),
    max_translation_frac: float = 0.15,
    p_flip: float = 0.5,
    perspective_jitter_frac: float = 0.04,
    elastic_magnitude: float = 25.0,
    elastic_smooth_sigma: float = 12.0,
    coord_per_pixel: float = 64.0 / 256.0,
    stop_grad_target: bool = True,
) -> torch.Tensor:
    """Equivariance consistency loss for an offset-prediction model.

    Parameters
    ----------
    model : callable
        Maps `(B, C_in, H, W) → (B, 2, H, W)` per-pixel (y, x) offsets.
    images : torch.Tensor of shape (B, C_in, H, W)
        Input batch. Each element gets its own random aug T.
    rng : torch.Generator, optional
        For reproducible augmentation sampling.
    transformation_intensity : float
        Single master knob (∈ [0, 1] in usual practice; 0 = identity, 1 = the
        named per-parameter values below). Multiplies all magnitude/probability
        parameters: rotation magnitude (or P(rotate) in axis-aligned mode),
        scale_range deviation from 1.0, max_translation_frac, p_flip,
        perspective_jitter_frac, and elastic_magnitude. The smoothing sigma and
        coord_per_pixel are unaffected. Default 0.1 = a gentle augmentation
        regime; pass 1.0 to recover the previous default behaviour.
    axis_aligned_rotation : bool
        If True, restrict rotation to {0, 90, 180, 270}°.
    scale_range : (float, float)
        Per-axis isotropic scale bounds. Independent y and x scales sampled per element.
    max_translation_frac : float
        Translation in fraction of `H`/`W`.
    p_flip : float
        Probability of horizontal/vertical flip independently.
    perspective_jitter_frac : float
        Corner-jitter magnitude as fraction of `H`.
    elastic_magnitude : float
        Raw amplitude of the unsmoothed random elastic field, in coord units.
        After Gaussian smoothing the effective per-pixel `|δ|` std is roughly
        `magnitude / (2π σ_px)`. Default 25 (coord units) ≈ old 100 (pixels).
    elastic_smooth_sigma : float
        Spatial smoothness of the elastic field (σ in pixels — the kernel
        operates on the H×W tensor grid, so this stays in pixel units).
    coord_per_pixel : float
        Conversion factor pixels → coord units. Default 64/256 matches
        InstanSeg's `generate_coordinate_map`. Pass `1.0` to operate purely
        in pixel space (legacy behaviour).
    stop_grad_target : bool
        If True (default), the original-frame model output is detached and
        acts as a fixed target — gradient only flows through the augmented
        branch. Required to prevent symmetric collapse to model ≡ 0 when the
        consistency term dominates the supervised signal (SimSiam-style).
        Set False for the symmetric formulation.

    Returns
    -------
    torch.Tensor (scalar)
        Mean squared error between `T_inv(model(T(x)))` and `model(x)`, masked
        to pixels where `T(p_orig)` lands inside the image bounds.
    """
    if images.dim() != 4:
        raise ValueError(f"images must be (B, C, H, W); got {tuple(images.shape)}")
    _, _, H, W = images.shape
    device = images.device

    if rng is None:
        rng = torch.Generator()

    # Apply the master intensity knob to all magnitude/probability parameters.
    # Per-param values represent the "intensity = 1" setting; intensity scales
    # them linearly (with scale_range interpolating between identity and the
    # asymmetric defaults).
    ti = float(transformation_intensity)
    eff_scale_range = (
        1.0 - (1.0 - scale_range[0]) * ti,
        1.0 + (scale_range[1] - 1.0) * ti,
    )
    eff_max_translation_frac = max_translation_frac * ti
    eff_p_flip = p_flip * ti
    eff_perspective_jitter_frac = perspective_jitter_frac * ti
    eff_elastic_magnitude = elastic_magnitude * ti

    # Coordinate grid in InstanSeg units, matching generate_coordinate_map():
    # linspace(0, H*coord_per_pixel, H). At default coord_per_pixel=64/256, the
    # canvas spans [0, H/4] in coord units; one coord unit ≈ 4 pixels.
    H_extent = H * coord_per_pixel
    W_extent = W * coord_per_pixel
    yy = torch.linspace(0, H_extent, H, device=device, dtype=torch.float32)
    xx = torch.linspace(0, W_extent, W, device=device, dtype=torch.float32)
    yy_g, xx_g = torch.meshgrid(yy, xx, indexing="ij")
    coord_grid = torch.stack([yy_g, xx_g], dim=0)  # (2, H, W) in (y, x) coord units

    # SimSiam-style asymmetry: the "target" is computed without grad, so the
    # consistency MSE only flows gradients through model(image_aug). This breaks
    # the symmetric collapse to model ≡ 0, where MSE(0, 0) = 0 minimizes both
    # sides simultaneously. The supervised loss anchors offset(orig) to be
    # nonzero; weight-sharing then propagates that signal via the aug branch.
    if stop_grad_target:
        with torch.no_grad():
            offset = model(images)
    else:
        offset = model(images)  # (B, 2, H, W) in coord units

    image_aug, unwarp = _apply_aug_and_make_unwarp(
        images, coord_grid, H_extent, W_extent, rng,
        axis_aligned_rotation=axis_aligned_rotation,
        scale_range=eff_scale_range,
        max_translation_frac=eff_max_translation_frac,
        p_flip=eff_p_flip,
        perspective_jitter_frac=eff_perspective_jitter_frac,
        elastic_magnitude=eff_elastic_magnitude,
        elastic_smooth_sigma=elastic_smooth_sigma,
        rotation_intensity=ti,
    )
    offset_aug = model(image_aug)
    inv_aug, valid_mask = unwarp(offset_aug)

    diff_sq = (inv_aug - offset) ** 2  # (B, 2, H, W)
    weights = valid_mask.float().unsqueeze(1).expand_as(diff_sq)
    return (diff_sq * weights).sum() / weights.sum().clamp_min(1.0)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch.nn as nn

    class TinyOffsetModel(nn.Module):
        """Trivial offset predictor for testing."""

        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 2, 3, padding=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.tanh(self.conv(x)) * 5.0

    torch.manual_seed(0)
    rng = torch.Generator().manual_seed(0)

    H = W = 64
    B = 4
    model = TinyOffsetModel()
    images = torch.rand(B, 1, H, W)

    loss = consistency_loss(model, images, rng=rng)
    print(f"cons loss = {loss.item():.4f}")
    print(f"requires_grad = {loss.requires_grad}")  # should be True

    # Backward sanity
    loss.backward()
    print(f"conv.weight grad norm = {model.conv.weight.grad.norm().item():.4f}")
