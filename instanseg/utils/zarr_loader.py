"""Manifest-driven, lazy zarr segmentation dataset + DataLoader factory.

Replaces ``data_loader._read_images_from_pth`` / ``get_loaders`` and the in-RAM
``Segmentation_Dataset``. Items are read lazily from the zarr layout produced by
``scripts.build_zarr_dataset``; scale selection, bounded cropping and global
percentile normalization happen in ``zarr_dataset.read_training_crop`` (so the
``normalize`` and ``torch_rescale`` augmentations are dropped from the pipeline),
and the remaining augmentations run on the small tile via the existing
``Augmentations`` class.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from instanseg.utils.augmentations import Augmentations
from instanseg.utils.augmentation_config import get_augmentation_dict
from instanseg.utils.AI_utils import collate_fn
from instanseg.utils.zarr_dataset import (
    TILE, open_item, sample_ds, read_training_crop, apply_normalization)

# Augmentations the reader now performs; removed from the per-tile pipeline.
_READER_ABSORBED = ("normalize", "torch_rescale")


def _trim_augmentation_dict(aug_dict: dict) -> dict:
    """Drop reader-absorbed augmentations from every modality OrderedDict."""
    for phase in aug_dict.values():            # 'train' / 'test'
        for modality_od in phase.values():     # per-modality OrderedDict
            for key in _READER_ABSORBED:
                modality_od.pop(key, None)
    return aug_dict


def _norm_src(s) -> str:
    """Normalize a source identifier so ``open_ai``/``open-ai`` and
    ``DeepBacs``/``deepbacs`` compare equal (mirrors the builder's ``_norm``)."""
    return str(s).lower().replace("-", "_")


def _source_filter(source_dataset) -> set[str] | None:
    if source_dataset in ("all", ["all"], None):
        return None
    if isinstance(source_dataset, str):
        return {_norm_src(source_dataset)}
    return {_norm_src(s) for s in source_dataset}


# A source token may carry a per-source sampling weight: ``name:0.1`` or
# ``name(0.1)``. The number is a relative frequency multiplier (default 1.0);
# weights need not sum to 1.
_SRC_WEIGHT_RE = re.compile(r"^(.*?)(?::|\()\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\)?$")


def _parse_source_token(tok: str) -> tuple[str, float | None]:
    tok = tok.strip().replace("'", "").replace('"', "")
    m = _SRC_WEIGHT_RE.match(tok)
    if m:
        return m.group(1).strip().lower(), float(m.group(2))
    return tok.lower(), None


def parse_source_dataset(raw) -> tuple[object, dict[str, float]]:
    """Split ``--source_dataset`` into (clean source spec, per-source weights).

    Accepts ``all`` | ``name`` | ``name:w`` | ``[n1, n2:w2, n3(w3)]`` (and an
    already-parsed list). Returns ``(source_dataset, weights)`` where
    ``source_dataset`` is what the rest of the pipeline expects (``"all"`` or a
    single lowercased name or a list of lowercased names, weights stripped), and
    ``weights`` maps ``_norm_src(name) -> float`` only for tokens that gave one.
    """
    if isinstance(raw, (list, tuple)):
        tokens, bracket = list(raw), True
    else:
        s = str(raw).strip()
        if "[" in s:
            tokens, bracket = s.replace("[", "").replace("]", "").split(","), True
        else:
            tokens, bracket = [s], False

    names: list[str] = []
    weights: dict[str, float] = {}
    for tok in tokens:
        if not str(tok).strip():
            continue
        name, w = _parse_source_token(str(tok))
        names.append(name)
        if w is not None:
            weights[_norm_src(name)] = w
    if not bracket:
        return (names[0] if names else "all"), weights
    return names, weights


def ensure_zarr_dataset(zarr_root: str | Path, data_dir: str | Path,
                        monolith_name: str = "segmentation_dataset.pth",
                        splits: tuple[str, ...] = ("Train", "Validation"),
                        accelerator=None, source_dataset="all") -> Path:
    """Return the manifest path, building only the requested ``source_dataset``.

    Source-scoped + change-aware: each source (a ``parent_dataset``) is written
    from its part file if one exists, else from the monolith, and is rebuilt
    automatically when that provenance ``.pth`` changes on disk (mtime/size). A
    source already built and unchanged is skipped, so this is cheap to call every
    run. The combined ``manifest.parquet`` is the union of all built sources;
    switching to a smaller ``--source_dataset`` does not delete previously built
    sources (they stay available, just unselected by the loader's filter).

    DDP-safe: only the main process builds; the others wait at a barrier. State
    is committed per source, so an interrupted build resumes cleanly next run.
    """
    from instanseg.scripts.build_zarr_dataset import incremental_build
    zarr_root = Path(zarr_root)
    is_main = accelerator is None or accelerator.is_main_process

    if is_main:
        if source_dataset in ("all", ["all"], None):
            requested = None
        elif isinstance(source_dataset, str):
            requested = [source_dataset]
        else:
            requested = list(source_dataset)
        incremental_build(out=zarr_root, data_dir=data_dir, monolith_name=monolith_name,
                          requested=requested, splits=list(splits), verbose=True)
    if accelerator is not None:
        accelerator.wait_for_everyone()
    return zarr_root / "manifest.parquet"


class ZarrSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_path: str | Path, split: str, args, is_train: bool):
        self.root = Path(manifest_path).parent
        self.split = split
        self.is_train = is_train
        self.target = args.target_segmentation
        self.requested_pixel_size = args.requested_pixel_size
        self.augmentation_type = args.augmentation_type
        self.tile = getattr(args, "tile_size", TILE)
        self.rng_seed = getattr(args, "rng_seed", None)
        self.force_pseudo = getattr(args, "force_pseudo_pixel_size", False)

        m = pd.read_parquet(manifest_path)
        m = m[m["split"] == split]
        srcs = _source_filter(getattr(args, "source_dataset", "all"))
        if srcs is not None:
            m = m[m["parent_dataset"].map(_norm_src).isin(srcs)]
        mod = getattr(args, "modality_filter", None)
        if mod:
            m = m[m["modality"].str.lower() == mod.lower()]
        # target availability mirrors data_loader._keep_images
        if self.target == "N":
            m = m[m["has_nucleus"]]
        elif self.target == "C":
            m = m[m["has_cell"]]
        else:  # NC: need at least one channel
            m = m[m["has_nucleus"] | m["has_cell"]]
        if len(m) == 0:
            raise ValueError(f"No items match split={split!r} target={self.target!r} "
                             f"source={getattr(args, 'source_dataset', 'all')!r}.")
        self.rows = m.reset_index(drop=True)
        self.parent_datasets = self.rows["parent_dataset"].tolist()

        aug = get_augmentation_dict(args.dim_in, nuclei_channel=None,
                                    amount=args.transform_intensity,
                                    pixel_size=args.requested_pixel_size,
                                    augmentation_type=args.augmentation_type,
                                    use_instance_channels=args.use_instance_channels)
        aug = _trim_augmentation_dict(aug)
        phase = "train" if is_train else "test"
        self.augmenter = Augmentations(augmentation_dict=aug[phase], debug=False,
                                       shape=(self.tile, self.tile), dim_in=args.dim_in,
                                       cells_and_nuclei=args.cells_and_nuclei,
                                       target_segmentation=args.target_segmentation,
                                       channel_invariant=args.channel_invariant,
                                       random_seed=args.rng_seed)
        self._handles: dict[str, tuple] = {}
        self._rng: np.random.Generator | None = None

    def __len__(self) -> int:
        return len(self.rows)

    def raw_view(self, idx: int):
        """Full-resolution normalized image + first label (numpy) for visualization."""
        row = self.rows.iloc[idx]
        group, attrs = self._open(row["rel_path"])
        img = apply_normalization(torch.from_numpy(np.asarray(group["0"])),
                                  attrs["perc_lo"], attrs["perc_hi"]).numpy()
        names = set(group["labels"].attrs.get("labels", []))
        lab = None
        for n in ("nucleus", "cell"):
            if n in names:
                lab = np.asarray(group[f"labels/{n}/0"])
                break
        return img, lab

    def _open(self, rel_path: str):
        # Per-worker handle cache (zarr groups are not fork-safe to share).
        h = self._handles.get(rel_path)
        if h is None:
            h = open_item(self.root / rel_path)
            self._handles[rel_path] = h
        return h

    def _rng_for(self, idx: int) -> np.random.Generator:
        if self.is_train:
            if self._rng is None:
                wi = torch.utils.data.get_worker_info()
                seed = (int(torch.initial_seed()) ^ ((wi.id + 1) if wi else 0)) & 0xFFFFFFFF
                self._rng = np.random.default_rng(seed)
            return self._rng
        # validation: stable per-item crops across epochs
        return np.random.default_rng(((self.rng_seed or 0) * 1000003 + idx) & 0xFFFFFFFF)

    def __getitem__(self, idx: int):
        row = self.rows.iloc[idx]
        group, attrs = self._open(row["rel_path"])
        rng = self._rng_for(idx)

        scale_attrs = {**attrs, "pixel_size": None} if self.force_pseudo else attrs
        ds = sample_ds(scale_attrs, self.target, self.requested_pixel_size,
                       self.augmentation_type, self.is_train, rng)
        image, label = read_training_crop(group, attrs, ds, self.tile, self.target, rng)

        # Only include optional keys when present — Augmentations branches on key
        # existence (e.g. np.random.choice(nuclei_channels)) and mishandles None.
        meta = {"image_modality": attrs["modality"],
                "pixel_size": attrs.get("pixel_size") or 1.0}  # float -> skip pseudo recompute
        if attrs.get("nuclei_channels"):
            meta["nuclei_channels"] = attrs["nuclei_channels"]
        if attrs.get("channel_names"):
            meta["channel_names"] = attrs["channel_names"]
        # Pass labels as numpy so to_tensor renumbers instance ids per channel
        # (skipping -1 sentinel channels), matching the legacy Segmentation_Dataset.
        image, label = self.augmenter(image, label.numpy(), meta)

        if label.dim() == 2:
            label = label[None]
        if image.dim() == 2:
            image = image[None]
        assert not torch.isnan(image).any(), "Transformed image contains NaN"
        assert not torch.isnan(label.float()).any(), "Transformed label contains NaN"
        return image.float(), label


def _make_sampler(parent_datasets: list[str], num_samples: int | None,
                  balance: bool, source_weights: dict[str, float] | None):
    """Per-item sampler weighting.

    ``balance`` (the ``-w`` flag) gives each source an equal share via
    inverse-frequency weighting. ``source_weights`` applies a per-source relative
    multiplier (default 1.0) on top — so with ``-w`` the source-level sampling
    probability is proportional to its multiplier (the per-source counts cancel),
    and without ``-w`` it scales the source's natural (size-proportional) share.
    Returns a ``WeightedRandomSampler`` when balancing or any multiplier is in
    play, else ``None`` (caller uses a plain ``RandomSampler`` — unchanged path).
    """
    from torch.utils.data import WeightedRandomSampler
    parents = np.asarray(parent_datasets)
    n = len(parents)
    if balance:
        names, counts = np.unique(parents, return_counts=True)
        cnt = dict(zip(names.tolist(), counts.tolist()))
        base = np.array([1.0 / cnt[p] for p in parents], dtype=np.float64)
    else:
        base = np.ones(n, dtype=np.float64)
    has_mult = bool(source_weights)
    if has_mult:
        base = base * np.array([float(source_weights.get(_norm_src(p), 1.0))
                                for p in parents], dtype=np.float64)
    if not balance and not has_mult:
        return None
    total = base.sum()
    if total <= 0:
        raise ValueError("All sampling weights are zero — check --source_dataset weights.")
    weights = torch.as_tensor(base / total, dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples or n, replacement=True)


def get_zarr_loaders(args):
    """Build (train_loader, test_loader) from ``args.manifest_path``."""
    from torch.utils.data import DataLoader, RandomSampler

    if args.rng_seed is not None:
        torch.manual_seed(args.rng_seed)

    manifest_path = args.manifest_path
    train_data = ZarrSegmentationDataset(manifest_path, "Train", args, is_train=True)
    val_data = ZarrSegmentationDataset(manifest_path, "Validation", args, is_train=False)

    balance = bool(getattr(args, "weight", False))
    source_weights = getattr(args, "source_weights", None)

    train_sampler = _make_sampler(train_data.parent_datasets, args.length_of_epoch,
                                  balance, source_weights)
    if train_sampler is None:
        train_sampler = (RandomSampler(train_data, num_samples=args.length_of_epoch)
                         if args.length_of_epoch is not None else RandomSampler(train_data))
    test_sampler = _make_sampler(val_data.parent_datasets, args.length_of_eval,
                                 balance, source_weights)
    if test_sampler is None:
        test_sampler = RandomSampler(val_data, num_samples=args.length_of_eval)

    loader_kwargs = {}
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = getattr(args, "prefetch_factor", 2)
    persistent = getattr(args, "persistent_workers", False)

    train_loader = DataLoader(train_data, collate_fn=collate_fn, batch_size=args.batch_size,
                              num_workers=args.num_workers, sampler=train_sampler,
                              persistent_workers=persistent, drop_last=True, **loader_kwargs)
    test_loader = DataLoader(val_data, collate_fn=collate_fn, batch_size=args.batch_size,
                             num_workers=args.num_workers, sampler=test_sampler,
                             persistent_workers=persistent, drop_last=True, **loader_kwargs)
    return train_loader, test_loader
