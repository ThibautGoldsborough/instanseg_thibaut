"""
Self-embedding extraction + Leiden clustering + UMAP visualization.

Called from train.py after hotstart when -cluster_sample is set. Hooks a named
tap module on the training model, runs eval-augmented train images through the
real forward pass N times (each pass gets a fresh random crop + flip from the
eval aug pipeline, averaged per image), and clusters the resulting embeddings
with Leiden.
"""

import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import scanpy as sc


def _img_to_thumbnail(img_tensor, size=48):
    """Convert an image array/tensor to a small RGB numpy thumbnail for plotting."""
    import torch.nn.functional as F
    from instanseg.utils.utils import _move_channel_axis

    if isinstance(img_tensor, np.ndarray):
        img_tensor = _move_channel_axis(img_tensor)  # ensure CHW
        img_tensor = torch.from_numpy(img_tensor.copy())
    img = img_tensor.float()
    C = img.shape[0]
    if C == 1:
        img = img.repeat(3, 1, 1)
    elif C > 3 or C == 2:
        # Deterministic linear projection to 3 channels
        gen = torch.Generator().manual_seed(42)
        colours = torch.rand(C, 3, generator=gen)
        img = torch.einsum("chw,ct->thw", img, colours)
    # Normalize to [0, 1]
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()
    # Resize to thumbnail
    img = F.interpolate(img.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False)[0]
    return img.permute(1, 2, 0).numpy()


def _plot_umap_thumbnails(adata, dataset, output_path, n_per_cluster=3):
    """Scatter UMAP with image thumbnails, subsampled n_per_cluster images per Leiden cluster."""
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    umap_coords = adata.obsm["X_umap"]
    cluster_labels = adata.obs["leiden"].values

    rng = np.random.default_rng(42)
    sampled_indices = []
    for cluster in np.unique(cluster_labels):
        idxs = np.where(cluster_labels == cluster)[0]
        chosen = rng.choice(idxs, size=min(n_per_cluster, len(idxs)), replace=False)
        sampled_indices.extend(chosen)

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.scatter(umap_coords[:, 0], umap_coords[:, 1], s=4, c="lightgrey", alpha=0.4)

    for idx in sampled_indices:
        raw_img = dataset.X[idx]
        thumb = _img_to_thumbnail(raw_img, size=48)
        im = OffsetImage(thumb, zoom=1.0)
        ab = AnnotationBbox(im, umap_coords[idx], frameon=True, pad=0.1,
                            bboxprops=dict(edgecolor="black", linewidth=0.5))
        ax.add_artist(ab)

    ax.set_title("UMAP with image thumbnails (subsampled per Leiden cluster)")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_aspect("equal")
    fig.savefig(output_path / "umap_thumbnails.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Thumbnail UMAP saved to {output_path / 'umap_thumbnails.png'}")


def _plot_cluster_grid(adata, dataset, output_path, n_cols=10, thumb_size=64):
    """Save an image grid: rows = Leiden clusters, columns = sample images per cluster."""
    import matplotlib.pyplot as plt

    cluster_labels = adata.obs["leiden"].values
    # Sort numerically; -1 (noise) comes first so it's visually distinct at the top.
    clusters = sorted(np.unique(cluster_labels), key=lambda x: int(x))

    n_rows = len(clusters)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.2, n_rows * 1.2))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    rng = np.random.default_rng(42)
    for row, cluster in enumerate(clusters):
        idxs = np.where(cluster_labels == cluster)[0]
        chosen = rng.choice(idxs, size=min(n_cols, len(idxs)), replace=False)
        for col in range(n_cols):
            ax = axes[row, col]
            ax.set_xticks([])
            ax.set_yticks([])
            if col < len(chosen):
                raw_img = dataset.X[chosen[col]]
                thumb = _img_to_thumbnail(raw_img, size=thumb_size)
                ax.imshow(thumb)
            else:
                ax.axis("off")
            if col == 0:
                ax.set_ylabel(f"C{cluster}", fontsize=8, rotation=0, labelpad=20, va="center")

    fig.suptitle("Samples per Leiden cluster", fontsize=12)
    fig.tight_layout(rect=[0.03, 0, 1, 0.97])
    save_path = output_path / "cluster_grid.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Cluster grid saved to {save_path}")


def _find_embedding_tap(model: torch.nn.Module) -> torch.nn.Module:
    """Locate a submodule exposing get_embedding_tap() anywhere in the model tree.

    Using a hook on the tap lets the embedder share the real forward pass, so any
    wrappers above (DataParallel, AdaptorNet, etc.) apply the same preprocessing
    training uses — no wrapper-specific code needed here.
    """
    for m in model.modules():
        get_tap = getattr(m, "get_embedding_tap", None)
        if callable(get_tap):
            return get_tap()
    raise AttributeError(
        "No submodule exposes get_embedding_tap(). "
        "Add one to your backbone (see EUPE.get_embedding_tap)."
    )


def _build_eval_dataset(train_dataset, args):
    """Wrap the same images as ``train_dataset`` with eval-mode augmentations.

    Eval augs are deterministic preprocessing (percentile normalize, resize to
    tile_size, channel shaping to dim_in) — no random flips/jitter — so
    embeddings are stable across runs and match what the model sees on
    validation data.
    """
    from instanseg.utils.AI_utils import Segmentation_Dataset
    from instanseg.utils.augmentation_config import get_augmentation_dict

    aug_dict = get_augmentation_dict(
        args.dim_in,
        nuclei_channel=None,
        amount=args.transform_intensity,
        pixel_size=args.requested_pixel_size,
        augmentation_type=args.augmentation_type,
    )
    return Segmentation_Dataset(
        train_dataset.X,
        train_dataset.Y,
        metadata=train_dataset.metadata,
        size=(args.tile_size, args.tile_size),
        augmentation_dict=aug_dict["test"],
        debug=False,
        dim_in=args.dim_in,
        cells_and_nuclei=args.cells_and_nuclei,
        random_seed=args.rng_seed,
        target_segmentation=args.target_segmentation,
        channel_invariant=args.channel_invariant,
    )


def run_sampling(args, train_loader, train_meta, device, model, n_tta_passes: int = 10):
    """Cluster train images using the training model itself, save to embeddings.pkl.

    Hooks the model's ``get_embedding_tap()`` submodule and runs the real
    forward pass on eval-augmented train images. Wrappers (DataParallel stripped,
    AdaptorNet kept) pass through naturally. Output is consumed by get_loaders
    when args.weight is set.
    """
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    full_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    tap = _find_embedding_tap(full_model)
    print(f"Using training model ({type(full_model).__name__}) for embeddings, "
          f"tap={type(tap).__name__}")

    # Eval-augmented sequential loader so embedding i ↔ dataset image i.
    # Workers parallelize the CPU-bound augs (percentile-normalize, rescale,
    # random crop); persistent_workers avoids worker respawn between TTA passes.
    from torch.utils.data import DataLoader
    eval_dataset = _build_eval_dataset(train_loader.dataset, args)
    num_workers = max(1, args.num_workers)
    seq_loader = DataLoader(
        eval_dataset, batch_size=train_loader.batch_size, shuffle=False,
        collate_fn=train_loader.collate_fn, num_workers=num_workers,
        persistent_workers=True,
    )

    print(f"Extracting embeddings ({n_tta_passes} TTA passes via eval aug pipeline) ...")
    was_training = full_model.training
    full_model.eval()

    captured: dict = {}
    def _hook(_mod, _inp, out):
        captured["out"] = out
    handle = tap.register_forward_hook(_hook)

    # Each pass through the loader re-samples random crop + flips + pixel-size
    # jitter from the eval aug pipeline, so averaging over passes gives
    # crop/flip-invariant embeddings aligned with the training distribution.
    pass_arrays = []
    try:
        with torch.no_grad():
            for pass_idx in range(n_tta_passes):
                pass_feats = []
                for batch in tqdm(seq_loader, desc=f"pass {pass_idx + 1}/{n_tta_passes}"):
                    images_batch = batch[0].to(device)
                    _ = full_model(images_batch)
                    feat = captured["out"]
                    while feat.ndim > 2:
                        feat = feat.mean(dim=-1)
                    pass_feats.append(feat.cpu().numpy())
                pass_arrays.append(np.concatenate(pass_feats, axis=0))
    finally:
        handle.remove()
        if was_training:
            full_model.train()

    embeddings = np.mean(np.stack(pass_arrays, axis=0), axis=0)
    print(f"Embeddings shape: {embeddings.shape}")

    parent_datasets = [
        (meta["parent_dataset"] if meta is not None and "parent_dataset" in meta else "unknown")
        for meta in train_meta
    ]

    embeddings = (embeddings - embeddings.mean(axis=0, keepdims=True)) / np.clip(
        embeddings.std(axis=0, keepdims=True), 1e-8, None
    )
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    embeddings = embeddings / norms

    print("Running Leiden clustering ...")
    adata = sc.AnnData(embeddings)
    adata.obs["parent_dataset"] = parent_datasets[:len(embeddings)]
    sc.pp.neighbors(adata, n_neighbors=15, use_rep="X", random_state=42)
    sc.tl.leiden(adata, resolution=2.5, random_state=42)
    cluster_labels = np.array(adata.obs["leiden"])
    print(f"Found {adata.obs['leiden'].nunique()} clusters.")

    print("Computing UMAP ...")
    sc.tl.umap(adata, random_state=42)

    sc.settings.figdir = str(output_path)
    sc.pl.umap(adata, color="leiden", save="_cluster.png", show=False)
    sc.pl.umap(adata, color="parent_dataset", save="_dataset.png", show=False)

    _plot_umap_thumbnails(adata, train_loader.dataset, output_path, n_per_cluster=10)
    _plot_cluster_grid(adata, train_loader.dataset, output_path, n_cols=10)
    print(f"UMAP plots saved to {output_path}/")

    save_path = output_path / "embeddings.pkl"
    with open(save_path, "wb") as f:
        pickle.dump({
            "embeddings": embeddings,
            "cluster_labels": cluster_labels,
            "parent_datasets": parent_datasets[:len(embeddings)],
            "metadata": train_meta,
        }, f)
    print(f"Embeddings saved to {save_path}")
