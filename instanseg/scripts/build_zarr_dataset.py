"""Convert the legacy ``.pth`` segmentation dataset(s) into the zarr layout.

Reads the monolithic ``segmentation_dataset.pth`` and/or the ``parts/*.pth``
files, and writes one zarr group per item under ``--out`` plus a
``manifest.parquet`` that drives sampling/filtering at train time. This is the
single remaining consumer of the old in-RAM format; everything downstream reads
zarr.

Examples
--------
    # Small prototype slices (Phase 1 validation)
    uv run python -m instanseg.scripts.build_zarr_dataset \
        --out instanseg/datasets/zarr --parts open_ai --limit 50
    uv run python -m instanseg.scripts.build_zarr_dataset \
        --out instanseg/datasets/zarr --datasets cellseg --limit 20

    # Full conversion
    uv run python -m instanseg.scripts.build_zarr_dataset --out instanseg/datasets/zarr --all
"""

from __future__ import annotations

import argparse
import os
import re
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from instanseg.utils.data_loader import get_image
from instanseg.utils.utils import _move_channel_axis
from instanseg.utils.augmentations import measure_average_instance_area
from instanseg.utils.zarr_dataset import (
    ItemMeta, resolve_modality, compute_percentiles, _n_levels,
    _label_2d_for_modality, write_item_zarr,
)

SPLIT_DIRNAME = {"Train": "Train", "Validation": "Validation", "Test": "Test"}
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe(s: str, maxlen: int = 48) -> str:
    """Filesystem-safe, length-capped name fragment. Some datasets (e.g.
    CellBinDB) carry pathological file_names (long underscore-number strings)
    that overflow the 255-byte filename limit; the idx prefix guarantees
    uniqueness so truncating the stem is safe."""
    out = _SAFE.sub("_", str(s)).strip("_") or "item"
    return out[:maxlen]


def _as_list(v) -> list | None:
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return [int(x) if isinstance(x, (np.integer,)) else x for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    return [v]


def _read_mask(item: dict, key: str) -> np.ndarray | None:
    if key not in item:
        return None
    return np.asarray(get_image(item[key])).squeeze()


def convert_item(item: dict, split: str, idx: int, out_root: Path) -> dict | None:
    """Convert one source item; returns its manifest row (or None if skipped)."""
    image = np.asarray(get_image(item["image"]))
    image_chw = np.ascontiguousarray(_move_channel_axis(image))
    c, h, w = image_chw.shape

    cell = _read_mask(item, "cell_masks")
    nucleus = _read_mask(item, "nucleus_masks")
    if nucleus is None:  # generic 'masks' is treated as nuclei (matches _format_labels)
        nucleus = _read_mask(item, "masks")
    if nucleus is None and cell is None:
        return None

    label2d = _label_2d_for_modality(nucleus, cell)
    modality = resolve_modality(image_chw, label2d, item.get("image_modality"))
    lo, hi = compute_percentiles(image_chw)

    med_nuc = float(measure_average_instance_area(nucleus)) if nucleus is not None else None
    med_cell = float(measure_average_instance_area(cell)) if cell is not None else None
    ps = item.get("pixel_size")
    ps = float(ps) if isinstance(ps, (int, float)) and ps else None

    parent = str(item.get("parent_dataset", "unknown"))
    stem = _safe(item.get("file_name") or Path(str(item.get("image", f"item{idx}"))).stem)
    rel_path = f"{_safe(parent)}/{SPLIT_DIRNAME[split]}/{idx:06d}_{stem}.zarr"

    meta = ItemMeta(
        rel_path=rel_path, parent_dataset=parent, split=split, modality=modality,
        height=int(h), width=int(w), channels=int(c), n_levels=_n_levels(h, w),
        has_nucleus=nucleus is not None, has_cell=cell is not None,
        pixel_size=ps, median_nucleus_area=med_nuc, median_cell_area=med_cell,
        nuclei_channels=_as_list(item.get("nuclei_channels")),
        channel_names=_as_list(item.get("channel_names")),
        file_name=item.get("file_name") or item.get("filename"),
        subset=item.get("subset"),
    )
    write_item_zarr(out_root / rel_path, image_chw, nucleus, cell, lo, hi, meta)
    return meta.manifest_row()


def load_sources(data_dir: Path, parts: list[str] | None, use_monolith: bool,
                 monolith_name: str) -> dict[str, list[dict]]:
    """Merge requested source .pth files into a {split: [items]} dict."""
    merged: dict[str, list[dict]] = {}

    def _merge(d: dict):
        for split, items in d.items():
            merged.setdefault(split, []).extend(items)

    if use_monolith:
        mono = data_dir / monolith_name
        if mono.exists():
            print(f"Loading monolith {mono} ...", flush=True)
            _merge(torch.load(mono, weights_only=False))
    parts_dir = data_dir / "parts"
    if parts is not None:
        for name in parts:
            p = parts_dir / (name if name.endswith(".pth") else f"{name}.pth")
            print(f"Loading part {p} ...", flush=True)
            _merge(torch.load(p, weights_only=False))
    return merged


def run_build(out: Path, data_dir: Path = Path("instanseg/datasets"),
              monolith_name: str = "segmentation_dataset.pth",
              all_sources: bool = True, use_monolith: bool | None = None,
              parts: list[str] | None = None, datasets: list[str] | None = None,
              splits: list[str] = ("Train", "Validation"), limit: int | None = None,
              verbose: bool = True) -> Path:
    """Convert selected .pth sources into the zarr layout under ``out`` and write
    ``out/manifest.parquet``. Returns the manifest path.

    The manifest is written only on success and last, so its presence is a
    reliable "build complete" marker for the bootstrap.
    """
    out = Path(out)
    data_dir = Path(data_dir)
    os.environ.setdefault("INSTANSEG_DATASET_PATH", str(data_dir.resolve()))

    if all_sources:
        parts = [p.stem for p in sorted((data_dir / "parts").glob("*.pth"))]
        use_monolith = True
    elif use_monolith is None:
        # Default source is the monolith; suppress only when parts are chosen.
        use_monolith = parts is None

    sources = load_sources(data_dir, parts, use_monolith, monolith_name)

    keep = {d.lower() for d in datasets} if datasets else None
    rows: list[dict] = []
    skipped = errors = 0
    for split in splits:
        items = sources.get(split, [])
        if keep is not None:
            items = [it for it in items if str(it.get("parent_dataset", "")).lower() in keep]
        if limit is not None:
            items = items[:limit]
        for idx, item in enumerate(tqdm(items, desc=f"{split:11s}", ncols=100, disable=not verbose)):
            try:
                row = convert_item(item, split, idx, out)
                if row is None:
                    skipped += 1
                else:
                    rows.append(row)
            except Exception:
                errors += 1
                print(f"\n[error] {split} item {idx} ({item.get('parent_dataset')}):\n{traceback.format_exc()}")

    if not rows:
        raise SystemExit("No items converted — check --parts/--datasets/--monolith selection.")

    manifest = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    tmp_path = out / "manifest.parquet.tmp"
    manifest_path = out / "manifest.parquet"
    manifest.to_parquet(tmp_path, index=False)
    tmp_path.replace(manifest_path)  # atomic publish
    if verbose:
        print(f"\nWrote {len(rows)} items ({skipped} skipped, {errors} errored).")
        print(f"Manifest: {manifest_path}")
        print(manifest.groupby(["split", "parent_dataset"]).size())
    return manifest_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path, help="Output dataset root.")
    ap.add_argument("--data-dir", type=Path, default=Path("instanseg/datasets"),
                    help="Where the .pth sources live (also sets INSTANSEG_DATASET_PATH for get_image).")
    ap.add_argument("--monolith-name", default="segmentation_dataset.pth")
    ap.add_argument("--all", action="store_true", help="Convert the monolith + every part.")
    ap.add_argument("--monolith", action="store_true", help="Include the monolithic pth.")
    ap.add_argument("--parts", nargs="*", default=None, help="Part names to include (stems under parts/).")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="Keep only these parent_dataset values (case-insensitive).")
    ap.add_argument("--splits", nargs="*", default=["Train", "Validation"])
    ap.add_argument("--limit", type=int, default=None, help="Max items per split (prototype).")
    args = ap.parse_args()

    run_build(out=args.out, data_dir=args.data_dir, monolith_name=args.monolith_name,
              all_sources=args.all,
              use_monolith=None if args.all else (args.monolith or args.parts is None),
              parts=args.parts, datasets=args.datasets, splits=args.splits, limit=args.limit)


if __name__ == "__main__":
    main()
