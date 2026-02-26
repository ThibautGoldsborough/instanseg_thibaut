"""
Embedding extraction + Leiden clustering + UMAP visualization.

Called from train.py when -sampling_mode is set. Reuses the existing
train_loader so no duplicate data loading is needed.
"""

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import scanpy as sc

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _to_3ch(images: torch.Tensor) -> torch.Tensor:
    """Convert (B, C, H, W) to (B, 3, H, W): repeat grayscale or deterministic linear projection for >3 channels."""
    C = images.shape[1]
    if C == 1:
        images = images.repeat(1, 3, 1, 1)
    elif C > 3:
        # Deterministic linear projection: assign a seeded RGB colour per channel
        gen = torch.Generator(device=images.device).manual_seed(42)
        colours = torch.rand(C, 3, device=images.device, generator=gen)  # (C, 3)
        # images: (B, C, H, W) -> einsum -> (B, 3, H, W)
        images = torch.einsum("bchw,ct->bthw", images, colours)
    return images


def _imagenet_normalize(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    return (images - mean) / std


def _pad_to_multiple(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    _, _, H, W = x.shape
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x


# ---- DINOv2 ----

def _load_dino(device):
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", pretrained=True)
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model


def _extract_dino_cls(model, images: torch.Tensor) -> np.ndarray:
    """Extract CLS token from DINOv2. Returns (B, embed_dim) numpy array."""
    patch_size = model.patch_size
    images = _pad_to_multiple(images, patch_size)
    B, C, H, W = images.shape

    tokens = model.patch_embed(images)
    cls_token = model.cls_token.expand(B, -1, -1)
    tokens = torch.cat([cls_token, tokens], dim=1)
    tokens = tokens + model.interpolate_pos_encoding(tokens, H, W)

    for blk in model.blocks:
        tokens = blk(tokens)

    tokens = model.norm(tokens)
    cls_features = tokens[:, 0, :]  # (B, embed_dim)
    return cls_features.cpu().numpy()


# ---- SAM ----

def _load_sam(device):
    from segment_anything.modeling.image_encoder import ImageEncoderViT

    model = ImageEncoderViT(
        img_size=1024,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        out_chans=256,
        use_abs_pos=False,
        use_rel_pos=True,
        rel_pos_zero_init=True,
        window_size=14,
        global_attn_indexes=(5, 11, 17, 23),
    )

    checkpoint_path = Path("~/.sam/sam_vit_l_0b3195.pth").expanduser()
    if checkpoint_path.exists():
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        sam_state = {}
        for k, v in state_dict.items():
            if k.startswith("image_encoder."):
                new_key = k[len("image_encoder."):]
                if new_key == "pos_embed":
                    continue
                sam_state[new_key] = v
        missing, unexpected = model.load_state_dict(sam_state, strict=False)
        if missing:
            print(f"SAM missing keys: {missing}")
        if unexpected:
            print(f"SAM unexpected keys: {unexpected}")
    else:
        print(f"SAM checkpoint not found at {checkpoint_path}, using random weights.")

    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model


def _extract_sam_gap(model, images: torch.Tensor) -> np.ndarray:
    """Extract global-average-pooled features from SAM neck. Returns (B, 256) numpy array."""
    images = _pad_to_multiple(images, 16)
    features = model(images)  # (B, 256, H/16, W/16) after neck
    gap = features.mean(dim=(-2, -1))  # (B, 256)
    return gap.cpu().numpy()


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
    elif C > 3:
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
    leiden_labels = adata.obs["leiden"].values

    # Sample n_per_cluster indices per cluster
    rng = np.random.default_rng(42)
    sampled_indices = []
    for cluster in np.unique(leiden_labels):
        idxs = np.where(leiden_labels == cluster)[0]
        chosen = rng.choice(idxs, size=min(n_per_cluster, len(idxs)), replace=False)
        sampled_indices.extend(chosen)

    fig, ax = plt.subplots(figsize=(14, 14))
    # Background scatter of all points
    ax.scatter(umap_coords[:, 0], umap_coords[:, 1], s=4, c="lightgrey", alpha=0.4)

    for idx in sampled_indices:
        raw_img = dataset.X[idx]  # (C, H, W) tensor, pre-augmentation
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
    """Save an image grid: rows = Leiden clusters, columns = sample images from each cluster."""
    import matplotlib.pyplot as plt

    leiden_labels = adata.obs["leiden"].values
    clusters = np.unique(leiden_labels)
    # Sort clusters numerically
    clusters = sorted(clusters, key=lambda x: int(x))

    n_rows = len(clusters)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.2, n_rows * 1.2))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    rng = np.random.default_rng(42)
    for row, cluster in enumerate(clusters):
        idxs = np.where(leiden_labels == cluster)[0]
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


def run_sampling(args, train_loader, train_meta, device):
    """Main entry point called from train.py when -sampling_mode is set."""

    # Parse mode string
    mode = args.sampling_mode.lower()
    if mode == "leiden_dino":
        backbone_name = "dino"
    elif mode == "leiden_sam":
        backbone_name = "sam"
    else:
        raise ValueError(f"Unknown sampling_mode: {args.sampling_mode}. Use 'leiden_dino' or 'leiden_sam'.")

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load backbone
    print(f"Loading {backbone_name} backbone ...")
    if backbone_name == "dino":
        backbone = _load_dino(device)
        extract_fn = _extract_dino_cls
    else:
        backbone = _load_sam(device)
        extract_fn = _extract_sam_gap

    # Extract embeddings in sequential order so embedding i corresponds to dataset image i
    from torch.utils.data import DataLoader
    seq_loader = DataLoader(train_loader.dataset, batch_size=train_loader.batch_size,
                            shuffle=False, collate_fn=train_loader.collate_fn, num_workers=0)

    print("Extracting embeddings ...")
    all_embeddings = []
    with torch.no_grad():
        for batch in tqdm(seq_loader):
            images_batch = batch[0].to(device)
            images_batch = _to_3ch(images_batch)
            images_batch = _imagenet_normalize(images_batch, device)
            emb = extract_fn(backbone, images_batch)
            all_embeddings.append(emb)

    embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"Embeddings shape: {embeddings.shape}")

    # Gather parent_dataset labels from train_meta
    parent_datasets = []
    for meta in train_meta:
        if meta is not None and "parent_dataset" in meta:
            parent_datasets.append(meta["parent_dataset"])
        else:
            parent_datasets.append("unknown")

    # L2-normalize embeddings (cosine similarity for kNN graph)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    embeddings = embeddings / norms

    # Leiden clustering
    print("Running Leiden clustering ...")
    adata = sc.AnnData(embeddings)
    adata.obs["parent_dataset"] = parent_datasets[:len(embeddings)]
    sc.pp.neighbors(adata, n_neighbors=15, use_rep="X", random_state=42)
    sc.tl.leiden(adata, resolution=2.5, random_state=42)
    print(f"Found {adata.obs['leiden'].nunique()} clusters.")

    # UMAP + plots
    print("Computing UMAP ...")
    sc.tl.umap(adata, random_state=42)

    sc.settings.figdir = str(output_path)
    sc.pl.umap(adata, color="leiden", save="_leiden.png", show=False)
    sc.pl.umap(adata, color="parent_dataset", save="_dataset.png", show=False)

    # UMAP with image thumbnails (subsample per cluster)
    _plot_umap_thumbnails(adata, train_loader.dataset, output_path, n_per_cluster=10)

    # Grid: rows = clusters, columns = 10 sample images
    _plot_cluster_grid(adata, train_loader.dataset, output_path, n_cols=10)

    print(f"UMAP plots saved to {output_path}/")

    # Save embeddings + cluster labels
    save_path = output_path / "embeddings.pkl"
    with open(save_path, "wb") as f:
        pickle.dump({
            "embeddings": embeddings,
            "leiden_labels": np.array(adata.obs["leiden"]),
            "parent_datasets": parent_datasets[:len(embeddings)],
            "metadata": train_meta,
        }, f)
    print(f"Embeddings saved to {save_path}")
