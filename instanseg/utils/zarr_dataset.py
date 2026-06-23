"""Zarr-backed segmentation dataset: shared build + read helpers.

This module replaces the legacy in-RAM ``.pth`` pipeline (``data_loader.py``).
Items are stored one-per-folder as OME-NGFF-style zarr v3 groups under

    <root>/<parent_dataset>/<split>/<stem>.zarr/

Each group holds a (conditional) image pyramid, full-resolution integer label
arrays, and all per-item metadata in ``.attrs`` (mirrored into a parquet
manifest at the dataset root for cheap sampling/filtering). Normalization is
*not* baked into pixels; instead per-channel percentile cut points are stored
and applied to the small crop at read time (mathematically identical to the old
``percentile_normalize`` over the whole image, but ~free).

Design notes live in the project CLAUDE.md.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from instanseg.utils.utils import _move_channel_axis, _estimate_image_modality
from instanseg.utils.augmentations import measure_average_instance_area

# --- format constants (build and read must agree) ------------------------------
TILE: int = 256                 # chunk/tile edge in pixels
PERCENTILE: float = 0.1         # matches augmentations.normalize -> percentile_normalize
NORM_EPSILON: float = 1e-3      # matches percentile_normalize epsilon
PYRAMID_THRESHOLD: int = 2 * TILE   # build a pyramid only when max(H,W) exceeds this
MIN_LEVEL_SIZE: int = TILE      # stop downsampling once the coarsest level <= this
MAX_SHARD_SPAN: int = 8 * TILE  # cap shard edge so a single shard file stays modest
DTYPE_LABEL = np.int32

MODALITY_BUCKETS = ("Brightfield", "phase-contrast", "Fluorescence")

# Expected instance areas (µm^2) for the pseudo pixel-size calibration; mirror
# augmentations.Augmentations.__call__ (nuclei ~35, cells ~105 ~= 3x larger).
EXPECTED_AREA_NUCLEUS = 35.0
EXPECTED_AREA_CELL = 105.0


# --- metadata ------------------------------------------------------------------
@dataclass
class ItemMeta:
    """Per-item metadata mirrored into both zarr ``.attrs`` and the manifest."""
    rel_path: str
    parent_dataset: str
    split: str
    modality: str
    height: int
    width: int
    channels: int
    n_levels: int
    has_nucleus: bool
    has_cell: bool
    pixel_size: float | None
    median_nucleus_area: float | None
    median_cell_area: float | None
    nuclei_channels: list[int] | None
    channel_names: list[str] | None
    file_name: str | None
    subset: str | None

    def manifest_row(self) -> dict[str, Any]:
        return asdict(self)


def resolve_modality(image_chw: np.ndarray, label_2d: np.ndarray,
                     meta_modality: str | None) -> str:
    """Collapse a raw ``image_modality`` (or absence thereof) to a bucket.

    Mirrors ``Augmentations.__call__``: Brightfield/Chromogenic -> "Brightfield",
    demoted to "phase-contrast" when the image is not 3-channel; anything else
    passes through (typically "Fluorescence"); missing metadata is estimated
    from intensity-under-mask via ``_estimate_image_modality``.
    """
    if meta_modality is not None:
        if meta_modality in ("Brightfield", "Chromogenic"):
            # min over the squeezed shape == channel count for our CHW arrays
            return "Brightfield" if min(image_chw.squeeze().shape) == 3 else "phase-contrast"
        return meta_modality
    return _estimate_image_modality(image_chw, label_2d)


def compute_percentiles(image_chw: np.ndarray) -> tuple[list[float], list[float]]:
    """Per-channel (p_lo, p_hi) at [PERCENTILE, 100-PERCENTILE] on the full image.

    Stored so the reader can reproduce ``percentile_normalize`` on a crop:
    ``(x - lo) / max(NORM_EPSILON, hi - lo)``.
    """
    img = image_chw.astype(np.float32, copy=False)
    lo, hi = [], []
    for c in range(img.shape[0]):
        p_min, p_max = np.percentile(img[c], [PERCENTILE, 100 - PERCENTILE])
        lo.append(float(p_min))
        hi.append(float(p_max))
    return lo, hi


def apply_normalization(crop_chw: torch.Tensor, lo: list[float], hi: list[float]) -> torch.Tensor:
    """Reproduce percentile_normalize on a crop using stored per-channel cut points.

    ``crop_chw`` channel axis must align with ``lo``/``hi`` (the stored channel
    order). Channels beyond the stored stats (e.g. synthesised by augmentation)
    are left untouched.
    """
    out = crop_chw.float().clone()
    n = min(out.shape[0], len(lo))
    for c in range(n):
        denom = max(NORM_EPSILON, hi[c] - lo[c])
        out[c] = (out[c] - lo[c]) / denom
    return out


def _n_levels(h: int, w: int) -> int:
    """Number of pyramid levels (>=1). Halving stops at MIN_LEVEL_SIZE."""
    maxdim = max(h, w)
    if maxdim <= PYRAMID_THRESHOLD:
        return 1
    levels = 1
    while maxdim > MIN_LEVEL_SIZE:
        maxdim //= 2
        levels += 1
    return levels


def build_image_pyramid(image_chw: np.ndarray, n_levels: int) -> list[np.ndarray]:
    """Halve repeatedly with antialiased bilinear downsampling in float, then cast
    back to the source dtype so every level shares the stored percentile stats."""
    levels = [image_chw]
    if n_levels == 1:
        return levels
    src_dtype = image_chw.dtype
    cur = torch.from_numpy(image_chw.astype(np.float32))[None]  # (1,C,H,W)
    for _ in range(n_levels - 1):
        h, w = cur.shape[-2] // 2, cur.shape[-1] // 2
        cur = F.interpolate(cur, size=(max(1, h), max(1, w)), mode="bilinear",
                            antialias=True, align_corners=False)
        arr = cur[0].numpy()
        if np.issubdtype(src_dtype, np.integer):
            info = np.iinfo(src_dtype)
            arr = np.clip(np.round(arr), info.min, info.max)
        levels.append(arr.astype(src_dtype))
    return levels


def _axis_chunk_shard(dim: int) -> tuple[int, int]:
    """(chunk, shard) for one spatial axis. chunk = min(TILE, dim); shard is a
    chunk-multiple covering min(dim, MAX_SHARD_SPAN) (zarr requires shard % chunk == 0)."""
    chunk = min(TILE, dim)
    span = min(dim, MAX_SHARD_SPAN)
    shard = int(np.ceil(span / chunk) * chunk)
    return chunk, shard


def _label_2d_for_modality(nucleus: np.ndarray | None, cell: np.ndarray | None) -> np.ndarray:
    lab = nucleus if nucleus is not None else cell
    return np.zeros((1, 1), DTYPE_LABEL) if lab is None else np.asarray(lab).squeeze()


# --- write ---------------------------------------------------------------------
def _blosc():
    import zarr
    return zarr.codecs.BloscCodec(cname="zstd", clevel=5, shuffle=zarr.codecs.BloscShuffle.shuffle)


def _write_array(group, name: str, arr: np.ndarray):
    """Create a sharded, compressed array sized to ``arr`` and fill it."""
    if arr.ndim == 3:  # (C, H, W) image
        c, h, w = arr.shape
        ch, sh = _axis_chunk_shard(h)
        cw, sw = _axis_chunk_shard(w)
        chunks, shards = (c, ch, cw), (c, sh, sw)
    else:              # (H, W) label
        h, w = arr.shape
        ch, sh = _axis_chunk_shard(h)
        cw, sw = _axis_chunk_shard(w)
        chunks, shards = (ch, cw), (sh, sw)
    za = group.create_array(name, shape=arr.shape, chunks=chunks, shards=shards,
                            dtype=arr.dtype, compressors=_blosc())
    za[:] = arr
    return za


def write_item_zarr(out_path: str | Path, image_chw: np.ndarray,
                    nucleus: np.ndarray | None, cell: np.ndarray | None,
                    lo: list[float], hi: list[float], meta: ItemMeta) -> None:
    """Write one item as a zarr v3 group: image pyramid + label arrays + attrs."""
    import zarr
    out_path = str(out_path)
    root = zarr.open_group(out_path, mode="w")

    pyramid = build_image_pyramid(image_chw, meta.n_levels)
    datasets = []
    for lvl, arr in enumerate(pyramid):
        _write_array(root, str(lvl), np.ascontiguousarray(arr))
        datasets.append({"path": str(lvl),
                         "coordinateTransformations": [{"type": "scale",
                                                        "scale": [1.0, float(2 ** lvl), float(2 ** lvl)]}]})

    labels_group = root.create_group("labels")
    label_names = []
    for name, lab in (("nucleus", nucleus), ("cell", cell)):
        if lab is None:
            continue
        lab2d = np.ascontiguousarray(np.asarray(lab).squeeze().astype(DTYPE_LABEL))
        _write_array(labels_group.create_group(name), "0", lab2d)
        label_names.append(name)

    root.attrs["instanseg"] = {**meta.manifest_row(), "perc_lo": lo, "perc_hi": hi}
    root.attrs["multiscales"] = [{
        "version": "0.4",
        "axes": [{"name": "c", "type": "channel"},
                 {"name": "y", "type": "space"},
                 {"name": "x", "type": "space"}],
        "datasets": datasets,
    }]
    labels_group.attrs["labels"] = label_names


# --- read ----------------------------------------------------------------------
def open_item(item_path: str | Path):
    """Open an item group read-only; returns (group, instanseg_attrs)."""
    import zarr
    g = zarr.open_group(str(item_path), mode="r")
    return g, dict(g.attrs["instanseg"])


def pick_level(ds: float, n_levels: int) -> int:
    """Largest pyramid level whose downsample (2**L) is <= the target downsample
    ``ds`` (so the read region is always downsampled to the tile, never upsampled
    from a too-coarse level)."""
    L = int(np.floor(np.log2(max(ds, 1.0))))
    return max(0, min(L, n_levels - 1))


def _resize_image(region: np.ndarray, out_hw: tuple[int, int]) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(region)).float()
    if t.shape[-2:] == torch.Size(out_hw):
        return t
    t = F.interpolate(t[None], size=out_hw, mode="bilinear", antialias=True, align_corners=False)
    return t[0]


def _resize_label(region: np.ndarray, out_hw: tuple[int, int]) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(region)).to(torch.int32)
    if t.shape[-2:] == torch.Size(out_hw):
        return t
    t = F.interpolate(t.float()[None, None], size=out_hw, mode="nearest")
    return t[0, 0].to(torch.int32)


def pseudo_pixel_size(median_nucleus_area: float | None,
                      median_cell_area: float | None, target: str) -> float | None:
    """Estimate µm/px from precomputed median instance areas, mirroring the
    per-item calibration in ``Augmentations.__call__``: prefer nuclei (~35 µm²)
    for the two-channel case, fall back to cells (~105 µm²); single target uses
    the matching expectation. Returns None when no usable area is available."""
    has_n = median_nucleus_area is not None and median_nucleus_area > 0
    has_c = median_cell_area is not None and median_cell_area > 0
    if "N" in target and "C" in target:
        if has_n:
            return float(np.sqrt(EXPECTED_AREA_NUCLEUS / median_nucleus_area))
        if has_c:
            return float(np.sqrt(EXPECTED_AREA_CELL / median_cell_area))
    elif target == "C":
        if has_c:
            return float(np.sqrt(EXPECTED_AREA_CELL / median_cell_area))
    else:  # "N"
        if has_n:
            return float(np.sqrt(EXPECTED_AREA_NUCLEUS / median_nucleus_area))
    return None


def sample_ds(attrs: dict, target: str, requested_pixel_size: float | None,
              augmentation_type: str, is_train: bool,
              rng: np.random.Generator) -> float:
    """Native-pixels-per-output-pixel downsample factor for one crop.

    Reproduces ``torch_rescale``'s pixel-size logic: no rescale (ds=1) when no
    target pixel size is requested or the source scale is unknown; otherwise
    ``ds = requested / current`` with ``requested`` jittered log-uniformly over a
    light (±10%) or heavy (0.25x–4x, train + heavy config only) range.
    """
    if requested_pixel_size is None:
        return 1.0
    current = attrs.get("pixel_size") or pseudo_pixel_size(
        attrs.get("median_nucleus_area"), attrs.get("median_cell_area"), target)
    if not current:
        return 1.0
    if augmentation_type in ("two_channel", "colourize", "brightfield_only"):
        lo_f, hi_f = 1.0, 1.0  # config uses range 0 -> fixed requested = ps
    elif augmentation_type == "heavy" and is_train:
        lo_f, hi_f = 0.25, 4.0
    else:
        lo_f, hi_f = 0.9, 1.1
    if lo_f == hi_f:
        requested = requested_pixel_size
    else:
        log_lo, log_hi = np.log(requested_pixel_size * lo_f), np.log(requested_pixel_size * hi_f)
        requested = float(np.exp(log_lo + rng.random() * (log_hi - log_lo)))
    return requested / current


def _pad_value(region: np.ndarray, modality: str) -> float:
    """Brightfield pads with the bright background (max); fluorescence/phase pad
    with the dark background (min) — matching ``torch_rescale``."""
    if region.size == 0:
        return 0.0
    return float(region.max() if modality == "Brightfield" else region.min())


def read_training_crop(group, attrs: dict, ds: float, tile: int, target: str,
                       rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Read one normalized ``(C,tile,tile)`` image crop + its label crop.

    Picks the coarsest pyramid level <= ``ds`` (bounded read), reads a randomly
    placed window covering ``tile*ds`` native px, pads out-of-bounds borders
    (image: modality background; labels: 0), resizes image (bilinear) and labels
    (nearest) to ``tile``, and percentile-normalizes the image from stored stats.

    The returned label is shaped to match ``data_loader._format_labels``: a
    ``(2,tile,tile)`` (nucleus, cell) stack for the two-channel target, otherwise
    ``(1,tile,tile)``. Absent channels are the all-``-1`` sentinel.
    """
    n_levels = int(attrs["n_levels"])
    lo, hi = attrs["perc_lo"], attrs["perc_hi"]
    modality = attrs["modality"]
    img0 = group["0"]
    C, H, W = img0.shape

    span = max(1, int(round(tile * ds)))
    top = int(rng.integers(0, H - span + 1)) if span < H else (H - span) // 2
    left = int(rng.integers(0, W - span + 1)) if span < W else (W - span) // 2

    # --- image: read at level L, place into a padded span_L buffer, resize ---
    L = pick_level(ds, n_levels)
    step = 2 ** L
    arr = group[str(L)]
    Hl, Wl = arr.shape[-2:]
    t0, l0 = top // step, left // step
    span_L = max(1, round(span / step))
    rt0, rl0 = max(0, t0), max(0, l0)
    rt1, rl1 = min(Hl, t0 + span_L), min(Wl, l0 + span_L)
    region = np.asarray(arr[:, rt0:rt1, rl0:rl1])
    region_n = apply_normalization(torch.from_numpy(np.ascontiguousarray(region)), lo, hi)
    buf = torch.full((C, span_L, span_L), _pad_value(region_n.numpy(), modality), dtype=torch.float32)
    buf[:, rt0 - t0: rt0 - t0 + region_n.shape[1], rl0 - l0: rl0 - l0 + region_n.shape[2]] = region_n
    image = _resize_image(buf.numpy(), (tile, tile))

    # --- labels: read the SAME native box the image level-L window covers, so
    # image and label stay registered (the image window is snapped to the level
    # grid: native [t0*step, (t0+span_L)*step)). Pad 0, nearest resize to tile. ---
    n_top, n_left, n_span = t0 * step, l0 * step, span_L * step
    label_names = set(group["labels"].attrs.get("labels", []))

    def _label_tile(name: str) -> torch.Tensor | None:
        if name not in label_names:
            return None
        la = group[f"labels/{name}/0"]
        lh, lw = la.shape
        b0, c0 = max(0, n_top), max(0, n_left)
        b1, c1 = min(lh, n_top + n_span), min(lw, n_left + n_span)
        lab = np.zeros((n_span, n_span), DTYPE_LABEL)
        lab[b0 - n_top: b1 - n_top, c0 - n_left: c1 - n_left] = np.asarray(la[b0:b1, c0:c1])
        return _resize_label(lab, (tile, tile))

    nuc, cell = _label_tile("nucleus"), _label_tile("cell")
    sentinel = torch.full((tile, tile), -1, dtype=torch.int32)
    if "N" in target and "C" in target:
        label = torch.stack([nuc if nuc is not None else sentinel,
                             cell if cell is not None else sentinel])
    elif target == "C":
        label = (cell if cell is not None else sentinel)[None]
    else:  # "N"
        label = (nuc if nuc is not None else sentinel)[None]
    return image, label


def read_crop_at_scale(group, attrs: dict, ds: float, tile: int,
                       top_native: int, left_native: int,
                       normalize: bool = True) -> dict[str, Any]:
    """Read a ``tile``x``tile`` output crop covering ``tile*ds`` native pixels
    starting at native (top, left).

    Picks the coarsest pyramid level <= ``ds`` for the image (bounded read,
    <= ~2*tile per side), reads the matching label region at full res, and
    resizes both to ``tile``. Image is percentile-normalized from stored stats.
    Assumes the requested native region lies within bounds (caller pads/clamps).
    """
    n_levels = int(attrs["n_levels"])
    lo, hi = attrs["perc_lo"], attrs["perc_hi"]
    span_native = int(round(tile * ds))

    L = pick_level(ds, n_levels)
    step = 2 ** L
    t0, l0 = top_native // step, left_native // step
    span_L = max(1, span_native // step)
    img_region = group[str(L)][:, t0:t0 + span_L, l0:l0 + span_L]
    image = _resize_image(img_region, (tile, tile))
    if normalize:
        image = apply_normalization(image, lo, hi)

    out: dict[str, Any] = {"image": image, "level": L}
    label_names = list(group["labels"].attrs.get("labels", []))
    for name in ("nucleus", "cell"):
        if name in label_names:
            reg = group[f"labels/{name}/0"][top_native:top_native + span_native,
                                            left_native:left_native + span_native]
            out[name] = _resize_label(reg, (tile, tile))
        else:
            out[name] = None
    return out
