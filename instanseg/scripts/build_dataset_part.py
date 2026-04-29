"""
Build a per-dataset .pth "part" that the training pipeline picks up at load time.

Each part is a {Train, Validation, Test} dict written into
<INSTANSEG_DATASET_PATH>/parts/<name>.pth. The data loader merges everything
in `parts/` on top of the legacy monolithic segmentation_dataset.pth, so
adding a new dataset never requires rewriting the main pth.

Usage:
    uv run python -m instanseg.scripts.build_dataset_part omnipose
    uv run python -m instanseg.scripts.build_dataset_part deepbacs
    uv run python -m instanseg.scripts.build_dataset_part omnipose --name omnipose_bact_only --subsets bact_phase bact_fluor
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from instanseg.utils import data_download
from instanseg.utils.data_download import get_processed_datasets_dir


LOADERS = {
    "omnipose": data_download.load_OmniPose,
    "deepbacs": data_download.load_DeepBacs,
    "usiigaci": data_download.load_Usiigaci,
    "vicar": data_download.load_Vicar,
    "toiam": data_download.load_TOIAM,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("loader", choices=sorted(LOADERS), help="Loader to run.")
    parser.add_argument("--name", default=None, help="Output filename stem (default: loader name).")
    parser.add_argument("--subsets", nargs="*", default=None,
                        help="Optional subset names forwarded to loaders that support filtering "
                             "(e.g. omnipose: bact_phase bact_fluor worm worm_high_res).")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    name = args.name or args.loader
    parts_dir = Path(get_processed_datasets_dir()) / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    out_path = parts_dir / f"{name}.pth"

    dataset: dict = {"Train": [], "Validation": [], "Test": []}
    loader = LOADERS[args.loader]

    kwargs = {"verbose": not args.quiet}
    if args.subsets is not None:
        kwargs["subsets" if args.loader == "omnipose" else "datasets"] = args.subsets

    loader(dataset, **kwargs)

    print(f"\nWriting part to {out_path}")
    print(f"  Train:      {len(dataset['Train'])}")
    print(f"  Validation: {len(dataset['Validation'])}")
    print(f"  Test:       {len(dataset['Test'])}")
    torch.save(dataset, out_path)
    print(f"Done. {out_path.stat().st_size / 1e6:.1f} MB on disk.")


if __name__ == "__main__":
    main()
