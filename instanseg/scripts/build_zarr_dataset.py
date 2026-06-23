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
import json
import os
import re
import shutil
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


def _norm(s: str) -> str:
    """Normalize a source identifier so part-file stems, ``parent_dataset`` values
    and user ``--source_dataset`` entries compare equal (``open_ai``/``open-ai``,
    ``DeepBacs``/``deepbacs``). Mirrors ``data_loader._read_images_from_pth``."""
    return str(s).lower().replace("-", "_")


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


# --- incremental, source-scoped build ----------------------------------------
# Only the requested sources are written, and a source is rebuilt automatically
# when its provenance .pth file changes on disk. State lives in build_state.json;
# each built source also has a manifest fragment under _fragments/, and the
# combined manifest.parquet is the union of all fragments.
STATE_NAME = "build_state.json"
FRAG_DIRNAME = "_fragments"


def resolve_sources(data_dir: Path, monolith_name: str) -> tuple[dict[str, Path], Path]:
    """Map normalized part-stem -> part path, and return the monolith path.

    A "source" (a ``parent_dataset``) is provided by ``parts/<stem>.pth`` when a
    matching part exists, otherwise by the monolith.
    """
    parts_dir = data_dir / "parts"
    parts_by_norm: dict[str, Path] = {}
    if parts_dir.is_dir():
        for p in sorted(parts_dir.glob("*.pth")):
            parts_by_norm[_norm(p.stem)] = p
    return parts_by_norm, data_dir / monolith_name


def _provenance(path: Path) -> dict:
    st = path.stat()
    return {"provenance": str(path), "mtime": st.st_mtime, "size": st.st_size}


def _provenance_changed(path: Path, mtime, size) -> bool:
    """True iff ``path`` exists and its (mtime, size) differ from the recorded pair.
    Unknown baseline or a missing file -> not changed (nothing to rebuild from)."""
    if mtime is None or size is None or not path.exists():
        return False
    st = path.stat()
    return st.st_mtime != mtime or st.st_size != size


def _state_path(out: Path) -> Path:
    return out / STATE_NAME


def _frag_path(out: Path, source: str) -> Path:
    return out / FRAG_DIRNAME / f"{source}.parquet"


def _load_state(out: Path) -> dict | None:
    sp = _state_path(out)
    if not sp.exists():
        return None
    try:
        return json.loads(sp.read_text()).get("sources", {})
    except Exception:
        return None


def _save_state(out: Path, state: dict) -> None:
    sp = _state_path(out)
    tmp = sp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"version": 1, "sources": state}, indent=2))
    tmp.replace(sp)


def _write_fragment(out: Path, source: str, df: pd.DataFrame) -> None:
    frag_dir = out / FRAG_DIRNAME
    frag_dir.mkdir(parents=True, exist_ok=True)
    fp = _frag_path(out, source)
    tmp = fp.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(fp)


def _monolith_sources(state: dict, monolith: Path) -> list[str]:
    return [s for s, v in state.items() if v.get("provenance") == str(monolith)]


def _clean_source(out: Path, state: dict, source: str) -> None:
    """Remove a source's zarr item folder + manifest fragment before a rebuild."""
    entry = state.get(source, {})
    folder = entry.get("folder")
    if folder:
        shutil.rmtree(out / folder, ignore_errors=True)
    _frag_path(out, source).unlink(missing_ok=True)


def _build_one_pth(pth_path: Path, out_root: Path, keep: set[str] | None,
                   splits: list[str], verbose: bool) -> dict[str, list[dict]]:
    """Convert one .pth, keeping items whose normalized parent_dataset is in
    ``keep`` (or all when ``keep`` is None). Returns {source_norm: [rows]}."""
    if verbose:
        print(f"Loading {pth_path} ...", flush=True)
    data = torch.load(str(pth_path), weights_only=False)
    rows_by_src: dict[str, list[dict]] = {}
    counters: dict[tuple, int] = {}
    for split in splits:
        items = data.get(split, [])
        for item in items:
            src = _norm(item.get("parent_dataset", "unknown"))
            if keep is not None and src not in keep:
                continue
            key = (src, split)
            idx = counters.get(key, 0)
            counters[key] = idx + 1
            try:
                row = convert_item(item, split, idx, out_root)
            except Exception:
                print(f"\n[error] {split} {src} item {idx}:\n{traceback.format_exc()}")
                continue
            if row is not None:
                rows_by_src.setdefault(src, []).append(row)
    return rows_by_src


def _migrate_from_manifest(out: Path, parts_by_norm: dict[str, Path],
                           monolith: Path) -> dict:
    """Bootstrap incremental state from a legacy full manifest.parquet (no state).

    Splits the existing manifest into per-source fragments and stamps each
    source's provenance with the *current* mtime/size — i.e. assumes the on-disk
    build matches the current .pth files (true right after a full build). This
    avoids rebuilding data that is already present; later changes are detected
    normally.
    """
    state: dict = {}
    mp = out / "manifest.parquet"
    if mp.exists():
        m = pd.read_parquet(mp)
        for parent, g in m.groupby("parent_dataset"):
            s = _norm(parent)
            p = parts_by_norm.get(s, monolith)
            _write_fragment(out, s, g.reset_index(drop=True))
            prov = _provenance(p) if p.exists() else {"provenance": str(p), "mtime": None, "size": None}
            state[s] = {**prov, "folder": str(g["rel_path"].iloc[0]).split("/")[0],
                        "n_items": int(len(g))}
        print(f"[zarr] migrated existing manifest into incremental state "
              f"({len(state)} sources).", flush=True)
    _save_state(out, state)
    return state


def _is_stale(out: Path, state: dict, source: str, prov_path: Path) -> bool:
    """A requested source needs (re)building if it was never built, its items or
    fragment are missing, or its provenance .pth changed."""
    entry = state.get(source)
    if entry is None:
        return True
    folder = entry.get("folder")
    if folder and not (out / folder).exists():
        return True
    if entry.get("n_items", 0) > 0 and not _frag_path(out, source).exists():
        return True
    return _provenance_changed(prov_path, entry.get("mtime"), entry.get("size"))


def _monolith_dirty(out: Path, state: dict, monolith: Path) -> bool:
    """For an ``all`` build: rebuild every monolith source if the monolith file
    changed, was never built, or any monolith source's items went missing."""
    if not monolith.exists():
        return False
    mono = {s: state[s] for s in _monolith_sources(state, monolith)}
    if not mono:
        return True
    v = next(iter(mono.values()))
    if _provenance_changed(monolith, v.get("mtime"), v.get("size")):
        return True
    for s, e in mono.items():
        folder = e.get("folder")
        if folder and not (out / folder).exists():
            return True
        if e.get("n_items", 0) > 0 and not _frag_path(out, s).exists():
            return True
    return False


def _rebuild_manifest(out: Path, state: dict) -> Path:
    """Concatenate all per-source fragments into manifest.parquet (atomic)."""
    frames = [pd.read_parquet(_frag_path(out, s)) for s in state
              if _frag_path(out, s).exists()]
    if not frames:
        raise SystemExit(f"No items built under {out} — check the source selection.")
    manifest = pd.concat(frames, ignore_index=True)
    tmp = out / "manifest.parquet.tmp"
    mp = out / "manifest.parquet"
    manifest.to_parquet(tmp, index=False)
    tmp.replace(mp)
    return mp


def incremental_build(out: Path, data_dir: Path = Path("instanseg/datasets"),
                      monolith_name: str = "segmentation_dataset.pth",
                      requested: list[str] | None = None,
                      splits: list[str] = ("Train", "Validation"),
                      verbose: bool = True) -> Path:
    """Build only the ``requested`` sources (None = all) into the zarr layout,
    (re)building a source whenever its provenance .pth changed. Returns the
    manifest path. Each source is loaded from its part file if one exists, else
    from the monolith. Safe to call repeatedly: unchanged sources are skipped.
    """
    out = Path(out)
    data_dir = Path(data_dir)
    os.environ.setdefault("INSTANSEG_DATASET_PATH", str(data_dir.resolve()))
    out.mkdir(parents=True, exist_ok=True)
    splits = list(splits)

    parts_by_norm, monolith = resolve_sources(data_dir, monolith_name)
    state = _load_state(out)
    if state is None:  # first run on this root (possibly a legacy full build)
        state = _migrate_from_manifest(out, parts_by_norm, monolith)

    requested_norm = None if requested is None else {_norm(s) for s in requested}

    # Plan: provenance .pth -> set of source norms to (re)build (None = every
    # source found in that file, used only for the monolith on an `all` build).
    plan: dict[Path, set[str] | None] = {}
    if requested_norm is None:
        for s, p in parts_by_norm.items():
            if _is_stale(out, state, s, p):
                plan.setdefault(p, set()).add(s)
        if _monolith_dirty(out, state, monolith):
            plan[monolith] = None
    else:
        for s in requested_norm:
            p = parts_by_norm.get(s, monolith)
            if _is_stale(out, state, s, p):
                plan.setdefault(p, set()).add(s)

    if not plan:
        if verbose:
            sel = "all sources" if requested_norm is None else f"{sorted(requested_norm)}"
            print(f"[zarr] {sel} already up to date at {out}", flush=True)
        return _rebuild_manifest(out, state)

    for pth, keep in plan.items():
        if not pth.exists():
            print(f"[zarr] skipping missing source file {pth}", flush=True)
            continue
        clean_targets = list(keep) if keep is not None else _monolith_sources(state, monolith)
        for s in clean_targets:
            _clean_source(out, state, s)
        if verbose:
            label = "ALL parent_datasets" if keep is None else sorted(keep)
            print(f"[zarr] building {label} from {pth.name} ...", flush=True)
        rows_by_src = _build_one_pth(pth, out, keep, splits, verbose)
        prov = _provenance(pth)
        for s, rows in rows_by_src.items():
            _write_fragment(out, s, pd.DataFrame(rows))
            state[s] = {**prov, "folder": rows[0]["rel_path"].split("/")[0],
                        "n_items": len(rows)}
        # Requested sources that yielded nothing: record state so we don't reload
        # the (possibly huge) provenance file again until it actually changes.
        if keep is not None:
            for s in keep - set(rows_by_src):
                state[s] = {**prov, "folder": state.get(s, {}).get("folder"), "n_items": 0}
        _save_state(out, state)

    return _rebuild_manifest(out, state)


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
