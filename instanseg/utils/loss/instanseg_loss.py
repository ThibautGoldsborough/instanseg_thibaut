import torch
import numpy as np
import pdb

from einops import rearrange
from typing import Tuple, List, Union
from instanseg.utils.pytorch_utils import torch_fastremap, torch_onehot,fast_sparse_intersection_over_minimum_area, remap_values, fast_iou, fast_sparse_iou, connected_components, flood_fill,fill_holes, expand_labels_map
from instanseg.utils.tiling import _instanseg_padding, _recover_padding

import torch.nn.functional as F
import torch.nn as nn

binary_xloss = torch.nn.BCEWithLogitsLoss()
l1_loss = torch.nn.L1Loss()

from instanseg.utils.utils import show_images
from instanseg.utils.utils import timer

integer_dtype = torch.int64


def convert(prob_input: torch.Tensor, coords_input: torch.Tensor, size: Tuple[int, int],
            mask_threshold: float = 0.5) -> torch.Tensor:
    # Create an array of labels for each pixel
    all_labels = torch.arange(1, 1 + prob_input.shape[0], dtype=torch.float32, device=prob_input.device)
    labels = torch.ones_like(prob_input) * torch.reshape(all_labels, (-1, 1, 1, 1))

    # Get flattened arrays
    labels = labels.flatten()
    prob = prob_input.flatten()
    x = coords_input[0, ...].flatten()
    y = coords_input[1, ...].flatten()

    # Predict image dimensions if we don't have them
    if size is None:
        size = (int(y.max() + 1), int(x.max() + 1))

    # Find indices with above-threshold probability values
    inds_prob = prob >= mask_threshold
    n_thresholded = torch.count_nonzero(inds_prob)
    if n_thresholded == 0:
        return torch.zeros(size, dtype=torch.float32, device=labels.device)

    # Create an array of [linear index, y, x, label], skipping low-probability values
    arr = torch.zeros((int(n_thresholded), 5), dtype=coords_input.dtype, device=labels.device)
    arr[:, 1] = y[inds_prob]
    arr[:, 2] = x[inds_prob]
    # NOTE: UNEXPECTED Y,X ORDER!
    arr[:, 0] = arr[:, 2] * size[1] + arr[:, 1]
    arr[:, 3] = labels[inds_prob]

    # Sort first by descending probability
    inds_sorted = prob[inds_prob].argsort(descending=True, stable=True)
    arr = arr[inds_sorted, :]

    # Stable sort by linear indices
    inds_sorted = arr[:, 0].argsort(descending=False, stable=True)
    arr = arr[inds_sorted, :]

    # Find the first occurrence of each linear index - this should correspond to the label
    # that has the highest probability, because they have previously been sorted
    inds_unique = torch.ones_like(arr[:, 0], dtype=torch.bool)
    inds_unique[1:] = arr[1:, 0] != arr[:-1, 0]

    # Create the output
    output = torch.zeros(size, dtype=torch.float32, device=labels.device)
    # NOTE: UNEXPECTED Y,X ORDER!
    output[arr[inds_unique, 2], arr[inds_unique, 1]] = arr[inds_unique, 3].float()

    return output


def find_all_local_maxima(image: torch.Tensor, neighbourhood_size: int, minimum_value: float) -> torch.Tensor:
    """
        helper function for peak_local_max that finds all the local maxima
        within each neighbourhood. (may return multiple per neighbourhood).
        """
    # Perform max pooling with the specified neighborhood size
    kernel_size = 2 * neighbourhood_size + 1
    pooled = F.max_pool2d(image, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    # Create a mask for the local maxima
    mask = (pooled == image) * (image >= minimum_value)

    # Apply the mask to the original image to retain only the local maxima values
    local_maxima = image * mask

    return local_maxima


def torch_peak_local_max(image: torch.Tensor, neighbourhood_size: int, minimum_value: float,
                             return_map: bool = False, dtype: torch.dtype = torch.int) -> torch.Tensor:
    """
    UPDATED FOR PERFORMANCE TESTING - NOT IDENTICAL, AS USES *FIRST* MAX, NOT FURTHEST FROM ORIGIN
    """
    h, w = image.shape
    image = image.view(1, 1, h, w)
    device = image.device

    kernel_size = 2 * neighbourhood_size + 1
    pooled, max_inds = F.max_pool2d(image, kernel_size=kernel_size, stride=1, padding=neighbourhood_size,
                                    return_indices=True)

    inds = torch.arange(0, image.numel(), device=device, dtype=dtype).reshape(image.shape)

    peak_local_max = (max_inds == inds) * (pooled > minimum_value)

    if return_map:
        return peak_local_max

    # Non-zero causes host-device synchronization, which is a bottleneck
    return torch.nonzero(peak_local_max.squeeze()).to(dtype)


def ensure_grid_maxima(seed_map, existing_maxima, grid_size=16):
    """Ensure at least one maximum exists in every grid_size x grid_size cell.

    Takes the existing local maxima and fills in any grid cell that has no
    maximum by picking the argmax of seed_map within that cell.

    Args:
        seed_map: [H, W] tensor — seed map values
        existing_maxima: [K, 2] int tensor — (y, x) of already-detected maxima
        grid_size: size of each square grid cell

    Returns:
        additional_maxima: [K', 2] int tensor — (y, x) of newly added maxima
    """
    H, W = seed_map.shape
    device = seed_map.device

    n_grid_y = (H + grid_size - 1) // grid_size
    n_grid_x = (W + grid_size - 1) // grid_size

    # Mark which grid cells already have a maximum
    occupied = torch.zeros(n_grid_y, n_grid_x, dtype=torch.bool, device=device)
    if existing_maxima.numel() > 0 and existing_maxima.shape[0] > 0:
        gy = (existing_maxima[:, 0] // grid_size).clamp(max=n_grid_y - 1).long()
        gx = (existing_maxima[:, 1] // grid_size).clamp(max=n_grid_x - 1).long()
        occupied[gy, gx] = True

    # Pad seed_map so it's evenly divisible by grid_size
    pad_h = n_grid_y * grid_size - H
    pad_w = n_grid_x * grid_size - W
    padded = F.pad(seed_map, (0, pad_w, 0, pad_h), value=float('-inf'))

    # Reshape into grid cells and find argmax within each
    cells = padded.reshape(n_grid_y, grid_size, n_grid_x, grid_size)
    cells = cells.permute(0, 2, 1, 3).reshape(n_grid_y, n_grid_x, grid_size * grid_size)
    flat_idx = cells.argmax(dim=-1)  # [n_grid_y, n_grid_x]

    local_y = flat_idx // grid_size
    local_x = flat_idx % grid_size

    # Convert to global coordinates
    grid_ys = torch.arange(n_grid_y, device=device).unsqueeze(1) * grid_size
    grid_xs = torch.arange(n_grid_x, device=device).unsqueeze(0) * grid_size
    global_y = (grid_ys + local_y).reshape(-1)
    global_x = (grid_xs + local_x).reshape(-1)

    # Keep only unoccupied cells with valid (non-padded) coordinates
    mask = ~occupied.reshape(-1) & (global_y < H) & (global_x < W)

    return torch.stack([global_y[mask], global_x[mask]], dim=1).int()


def torch_peak_local_max_LEGACY(image: torch.Tensor, neighbourhood_size: int, minimum_value: float, return_map: bool = False) -> torch.Tensor:
    """
    computes peak local maxima function for an image (or batch of images), returning a maxima mask
    and the coordinates of the peak local max values.
    peak local maxima returns a image that is zero at all points other than local maxima.
    At the local maxima, the pixel retains its value in the original image.
    
    image: a torch tensor of shape [B,1,H,W] or [H,W], B is batch size. H,W are spatial dims.
    neighbourhood_size: int. Only one maximum will be selected within a square patch of width
        equal to the neighbourhood size (specifically the largest maxima in that neighbourhood).
        Where there are multiple local maxima with the largest value within the neighbourhood,
        the maxima furthest away from the origin (furthest by euclidian distance from pixel (0,0))
        is retained (ensuring there is only one maximum per neighbourhood).
    minimum_value: float. Local maxima with pixel intensity below this value are ignored.
    
    returns: a torch tensor of shape equal to image, a list of length B containing (lx, ly) pairs
    where lx and ly are torch tensors containing the x and y coordinates of each local maxima for a given image.
    if image has shape [H,W], returns (lx, ly). 
    """
    assert image.ndim == 2, "image must be of shape [H,W]"

    h, w = image.shape
    image = image.view(1, 1, h, w)
    device = image.device

    all_local_maxima = find_all_local_maxima(image, neighbourhood_size, minimum_value)

    # perform non-maximal coordinate suppression to only get one maximum per neighbourhood.
    # specifically, where there are two maxima in a neighbourhood, I retain the maxima
    # which has the furthest euclidian distance away from the origin. This is just an
    # 'arbitrary' way for me to split the ties. 
    spatial_dims = [image.shape[-2], image.shape[-1]]

    grid = torch.stack(
        torch.meshgrid(
            torch.arange(0, spatial_dims[0], 1, device=device, dtype = torch.float32), torch.arange(0, spatial_dims[1], 1, device=device, dtype = torch.float32),
            indexing='ij'
        )
    )

    distance_to_origin = (grid.unsqueeze(0)).square().sum(dim=1).sqrt()

    distance_of_max_poses = torch.mul(all_local_maxima, distance_to_origin)

    retained_maxima = find_all_local_maxima(distance_of_max_poses, neighbourhood_size, minimum_value=minimum_value)
    peak_local_max = all_local_maxima * (retained_maxima > minimum_value)

    if return_map:
        return peak_local_max

    locs = grid[:,peak_local_max.squeeze()>0].T.int()


    return locs

#@torch.jit.script
def centre_crop(centroids: torch.Tensor, window_size: int, h:int, w:int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    centres the crop around the centroid, ensuring that the crop does not exceed the image dimensions.
    """
    C = centroids.shape[0]
    centroids = centroids.clone()  # C,2
    centroids[:, 0] = centroids[:, 0].clamp(min=window_size //2 , max=h - window_size //2)
    centroids[:, 1] = centroids[:, 1].clamp(min=window_size //2, max=w - window_size //2)
    window_slices = (centroids[:, None] + torch.tensor([[-1, -1], [1, 1]], device=centroids.device) * (window_size // 2))

    grid_x, grid_y = torch.meshgrid(
        torch.arange(window_size, device=centroids.device, dtype=centroids.dtype),
        torch.arange(window_size, device=centroids.device, dtype=centroids.dtype),
        indexing="ij"
    )

    mesh = torch.stack((grid_x, grid_y))
    mesh_grid = mesh.expand(C, 2, window_size, window_size)
    mesh_grid_flat = torch.flatten(mesh_grid, 2).permute(1, 0, 2)
    mesh_grid_flat = mesh_grid_flat + window_slices[:, 0].permute(1, 0)[:, :, None]
    mesh_grid_flat = torch.flatten(mesh_grid_flat, 1)

    return mesh_grid_flat, window_slices


def compute_crops( x: torch.Tensor, 
                  c: torch.Tensor, 
                  sigma: torch.Tensor,
                  centroids_idx: torch.Tensor,
                  feature_engineering, 
                  pixel_classifier, 
                  mask_threshold = None,
                  cleanup_fragments = False,
                  window_size: int = 128):

    h, w = x.shape[-2:]
    C = c.shape[0]

    mesh_grid_flat,window_slices = centre_crop(centroids_idx, window_size, h, w)

    x = feature_engineering(x, c, sigma, window_size //2 , mesh_grid_flat)

    x = pixel_classifier(x, seed_emb=c, window_size=window_size)  # C*H*W,1

    x = x.view(C, 1, window_size, window_size)
    idx = torch.arange(1, C + 1, device=x.device, dtype = mesh_grid_flat.dtype)

    rep = torch.ones((C, window_size, window_size), device=x.device, dtype =mesh_grid_flat.dtype)
    rep = rep * (idx[:, None, None] - 1)

    iidd = torch.cat((rep.flatten()[None,], mesh_grid_flat)).to(mesh_grid_flat.dtype)

    if cleanup_fragments and mask_threshold is not None:

        top_left = window_slices[:,0,:]
        shifted_centroid = centroids_idx - top_left

        seeds = torch.zeros_like(x)
        seeds[torch.arange(C, device=x.device), 0, shifted_centroid[:,0], shifted_centroid[:,1]] = 1

        filled = flood_fill((torch.sigmoid(x) >= mask_threshold), seeds) > 0
        holes = torch.bitwise_xor(fill_holes(filled), filled)

        x[~filled] = 0 #remove fragments
        x[holes] = mask_threshold #fill holes

    return x, iidd



def find_connected_components_legacy(adjacency_matrix: torch.Tensor, num_iterations: int = 10) -> torch.Tensor:

    M = (adjacency_matrix + torch.eye(adjacency_matrix.shape[0],
                               device=adjacency_matrix.device))  # https://math.stackexchange.com/questions/1106870/can-i-find-the-connected-components-of-a-graph-using-matrix-operations-on-the-gr
    num_iterations = 10#10

    out = torch.matrix_power(M, num_iterations)

    if torch.isinf(out).any() or torch.isnan(out).any():
        print("Warning: overflow detected in adjacency matrix. Too many seeds detected")

    
    col = torch.arange(0, out.shape[0], device=out.device).view(-1, 1).expand(out.shape[0], out.shape[
        0])  # Just a column matrix with numbers from 0 to out.shape[0]
    out_col_idx = ((out > 1).int() - torch.eye(out.shape[0], device=out.device)) * col
    maxes = out_col_idx.argmax(0) * (out_col_idx.max(0)[0] > 0).int()
    maxes = torch.maximum(maxes + 1, (torch.arange(0, out.shape[0],
                                                    device=out.device) + 1))  # recover the diagonal elements that were suppressed
    tentative_remapping = torch.stack(((torch.arange(0, out.shape[0], device=out.device) + 1), maxes))
    # start with two zeros:
    remapping = torch.cat((torch.zeros(2, 1, device=tentative_remapping.device), tentative_remapping),
                            dim=1)  # Maybe this can be avoided in the future by thresholding labels
    
    return remapping
@torch.jit.script
def find_connected_components(adjacency_matrix: torch.Tensor, max_iterations: int = 100) -> torch.Tensor:
    """
    Find connected components using label propagation, compatible with TorchScript.
    Args:
        adjacency_matrix: Binary square tensor (n x n) representing the graph.
        max_iterations: Maximum number of iterations to prevent infinite loops.
    Returns:
        remapping: Tensor (2 x (n+1)) where remapping[1][i] is the component label for node i (1-based).
    """
    # Validate input
    if not torch.all((adjacency_matrix == 0) | (adjacency_matrix == 1)):
        raise ValueError("Adjacency matrix must be binary (0s and 1s)")
    if not torch.all(adjacency_matrix == adjacency_matrix.t()):
        raise ValueError("Adjacency matrix must be symmetric (undirected graph)")
    n = adjacency_matrix.shape[0]
    if n == 0:
        return torch.zeros(2, 1, device=adjacency_matrix.device, dtype=torch.long)

    # Initialize labels as node indices (1-based)
    labels = torch.arange(1, n + 1, device=adjacency_matrix.device, dtype=torch.long)
    M = adjacency_matrix + torch.eye(n, device=adjacency_matrix.device)  # Include self-loops

    # Get non-zero indices as [num_nonzero, 2] tensor
    indices = torch.nonzero(M)  # Returns [row, col] pairs
    if indices.size(0) == 0:
        raise ValueError("Graph has no edges or self-loops; check input adjacency matrix")
    row = indices[:, 0]
    col = indices[:, 1]

    # Label propagation
    for i in range(max_iterations):
        prev_labels = labels.clone()
        min_labels = torch.full((n,), float('inf'), device=adjacency_matrix.device, dtype=torch.float)
        min_labels.scatter_reduce_(0, row, labels[col].float(), reduce='amin')
      #  if torch.any(~torch.isfinite(min_labels)):
        #    raise RuntimeError("Non-finite values detected in label propagation; possible numerical issue")
        new_labels = min_labels#.long()
       # if torch.any(new_labels > torch.iinfo(torch.int64).max) or torch.any(new_labels < torch.iinfo(torch.int64).min):
         #   raise RuntimeError("Label values exceed int64 range during conversion")
        labels = torch.minimum(labels, new_labels)
        if torch.equal(labels, prev_labels):
            # print(f"Converged after {i+1} iterations")  # print not supported in TorchScript
            break
        if i == max_iterations - 1:
            print(f"Warning: Maximum iterations ({max_iterations}) reached without convergence.")
            pass  # Avoid print in TorchScript

    # Create remapping tensor
    node_indices = torch.arange(1, n + 1, device=adjacency_matrix.device, dtype=torch.long)
    tentative_remapping = torch.stack((node_indices, labels))
    remapping = torch.cat((torch.zeros(2, 1, device=adjacency_matrix.device, dtype=torch.long), tentative_remapping), dim=1)

    return remapping

def has_pixel_classifier_model(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Module):
            module_class = module.__class__.__name__
            if 'ProbabilityNet' in module_class or 'AttentionPixelClassifier' in module_class:
                return True
    return False




def merge_sparse_predictions(x: torch.Tensor,
                             coords: torch.Tensor,
                               mask_map: torch.Tensor,
                               size : list[int],
                               mask_threshold: float = 0.5,
                               window_size = 128,
                                min_size = 10,
                                overlap_threshold = 0.5,
                                mean_threshold = 0.5,
                                seed_affinity: torch.Tensor = None,
                                overlap_metric: str = "iou"):


    labels = convert(x, coords, size=(size[1], size[2]), mask_threshold=mask_threshold)[None]



    idx = torch.arange(1, size[0] + 1, device=x.device, dtype =coords.dtype)
    stack_ID = torch.ones((size[0], window_size, window_size), device=x.device, dtype=coords.dtype)
    stack_ID = stack_ID * (idx[:, None, None] - 1)

    coords = torch.stack((stack_ID.flatten(), coords[0] * size[2] + coords[1])).to(coords.dtype)

    fg = x.flatten() > mask_threshold
    x = x.flatten()[fg]
    coords = coords[:, fg]

    using_mps = False
    if x.is_mps:
        using_mps = True
        device = 'cpu'
        x = x.to(device)
        mask_map = mask_map.to(device)

    sparse_onehot = torch.sparse_coo_tensor(
        coords,
        x.flatten() > mask_threshold,
        size=(size[0], size[1] * size[2]),
        dtype=x.dtype,
        device=x.device,
        requires_grad=False,
        )

    object_areas = torch.sparse.sum(sparse_onehot, dim=1).values()

    sum_mask_value = torch.sparse.sum((sparse_onehot * mask_map.flatten()[None]), dim=1).values()
    mean_mask_value = sum_mask_value / object_areas
    objects_to_remove = ~torch.logical_and(mean_mask_value > mean_threshold, object_areas > min_size)

    if window_size **2 * sparse_onehot.shape[0] == sparse_onehot.sum():
        #This can happen at the start of training. This can cause OOM errors and is never a good sign - may aswell abort.
        return labels

    if overlap_metric == "iomin":
        iou = fast_sparse_intersection_over_minimum_area(sparse_onehot)
    else:
        iou = fast_sparse_iou(sparse_onehot)

    adjacency = (iou > overlap_threshold)
    if seed_affinity is not None:
        sa = seed_affinity.to(adjacency.device)
        sym_affinity = torch.maximum(sa, sa.T)
        adjacency = adjacency | (sym_affinity > mask_threshold)

    remapping = find_connected_components(adjacency.float())

    if using_mps:
        device = 'mps'
        remapping = remapping.to(device)
        labels = labels.to(device)

    labels = remap_values(remapping, labels)

    labels_to_remove = (torch.arange(0, len(objects_to_remove), device=objects_to_remove.device, dtype = coords.dtype) + 1)[
    objects_to_remove]

    labels[torch.isin(labels, labels_to_remove)] = 0

    return labels

def guide_function(params: torch.Tensor,device ='cuda', width: int = 256):

    #params must be depth,3  

    depth = params.shape[0]
    xx = torch.linspace(0, 1, width, device=device).view(1, 1, -1).expand(1, width,width)
    yy = torch.linspace(0, 1, width, device=device).view(1, -1, 1).expand(1, width, width)
    xxyy  = torch.cat((xx, yy), 0).expand(depth,2,width,width)

    xx = xxyy[:,0] * params[:,0][:,None,None]
    yy = xxyy[:,1] * params[:,1][:,None,None]

    return torch.sin(xx+yy+params[:,2,None,None])[None]


def generate_coordinate_map(mode: str = "linear", spatial_dim: int = 2, height: int = 256, width: int = 256, device: torch.device = torch.device(type='cuda')):

    if mode == "rope":
        # RoPE needs raw linear coordinates — rotary encoding is applied
        # inside the attention mechanism, not in the coordinate map.
        if spatial_dim == 2:
            xx = torch.linspace(0, width * 64 / 256, width, device=device).view(1, 1, -1).expand(1, height, width)
            yy = torch.linspace(0, height * 64 / 256, height, device=device).view(1, -1, 1).expand(1, height, width)
            return torch.cat((xx, yy), 0)
        else:
            raise NotImplementedError(f"RoPE coordinate map not implemented for spatial_dim={spatial_dim}")

    if mode == "linear":
        if spatial_dim ==2:
            xx = torch.linspace(0, width * 64 / 256, width, device=device).view(1, 1, -1).expand(1, height,width)
            yy = torch.linspace(0, height * 64 / 256, height, device=device).view(1, -1, 1).expand(1, height, width)
            xxyy = torch.cat((xx, yy), 0)
            return xxyy
            
        elif spatial_dim == 3:

            # Pseudo-3D for 2D images: randomly select orthogonal plane
            # Randomly choose which coordinate to keep constant (0=x, 1=y, 2=z)
            plane_choice = torch.randint(0, 3, (1,), device=device).item()

            # Generate base coordinates for the 2D slice
            xx_2d = torch.linspace(0, width * 64 / 256, width, device=device).view(1, -1).expand(height, width)
            yy_2d = torch.linspace(0, height * 64 / 256, height, device=device).view(-1, 1).expand(height, width)

            # Choose a random constant value for the fixed dimension (between 0 and 64)
            const_val = torch.rand(1, device=device).item() * 64
            const_2d = torch.full((height, width), const_val, device=device)

            if plane_choice == 0:
                # YZ plane: x is constant, y and z vary
                xx = const_2d
                yy = xx_2d
                zz = yy_2d
            elif plane_choice == 1:
                # XZ plane: y is constant, x and z vary
                xx = xx_2d
                yy = const_2d
                zz = yy_2d
            else:  # plane_choice == 2
                # XY plane: z is constant, x and y vary
                xx = xx_2d
                yy = yy_2d
                zz = const_2d

            # Stack into (3, H, W)
            xxyy = torch.stack((xx, yy, zz), dim=0)
            return xxyy

        else:
            xxyy = torch.zeros((spatial_dim, height, width), device=device) #NOT IMPLEMENTED - THIS IS JUST A DUMMY VALUE

    else:
        xxyy = torch.zeros((spatial_dim, height, width), device=device) #NOT IMPLEMENTED - THIS IS JUST A DUMMY VALUE

    return xxyy


def precompute_rope_freqs_2d(head_dim, theta=10000.0):
    """Precompute rotation frequencies for 2D RoPE.
    head_dim split in half: first half for y, second half for x."""
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, 2).float() / half))
    return freqs  # shape: [half // 2]


def apply_rope_2d(x, positions, freqs):
    """Apply 2D rotary position embeddings.
    x: [*, num_heads, seq_len, head_dim]
    positions: [*, seq_len, 2] — (y, x) relative coords
    freqs: [quarter_dim] — precomputed frequencies
    """
    head_dim = x.shape[-1]
    half = head_dim // 2
    quarter = half // 2

    # Split into y-half and x-half
    x_y = x[..., :half]
    x_x = x[..., half:]

    # Compute angles: pos * freq for each pair
    pos_y = positions[..., 0:1]  # [*, seq_len, 1]
    pos_x = positions[..., 1:2]  # [*, seq_len, 1]

    # freqs: [quarter] -> angles: [*, seq_len, quarter]
    angles_y = pos_y * freqs.to(x.device)  # [*, seq_len, quarter]
    angles_x = pos_x * freqs.to(x.device)

    # Expand for num_heads dimension
    while angles_y.dim() < x_y.dim():
        angles_y = angles_y.unsqueeze(-3)
        angles_x = angles_x.unsqueeze(-3)

    # Rotate pairs (2i, 2i+1) using sin/cos
    cos_y, sin_y = angles_y.cos(), angles_y.sin()
    cos_x, sin_x = angles_x.cos(), angles_x.sin()

    def rotate_pairs(t, cos_a, sin_a):
        t1 = t[..., 0::2]
        t2 = t[..., 1::2]
        return torch.stack([t1 * cos_a - t2 * sin_a, t1 * sin_a + t2 * cos_a], dim=-1).flatten(-2)

    x_y_rot = rotate_pairs(x_y, cos_y, sin_y)
    x_x_rot = rotate_pairs(x_x, cos_x, sin_x)

    return torch.cat([x_y_rot, x_x_rot], dim=-1)


class ProbabilityNet(nn.Module):
    def __init__(self, embedding_dim=4, width = 5):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, 1)

    def forward(self, x, **kwargs):
        # x is C*H*W,E+S+1 (H,W is the window of the crop used here, e.g 100x100, not the image)
      #  with torch.cuda.amp.autocast():
        x = self._relu_non_empty(self.fc1(x))
        x = self._relu_non_empty(self.fc2(x))
        x = self.fc3(x)
        return x

    def _relu_non_empty(self, x: torch.Tensor) -> torch.Tensor:
        """
        Workaround for https://github.com/pytorch/pytorch/issues/118845 on MPS
        """
        if x.numel() == 0:
            return x
        else:
            return torch.relu_(x)



class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention block with feed-forward network."""
    def __init__(self, hidden_dim, n_heads, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        mlp_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, q, kv):
        q_norm = self.norm1(q)
        kv_norm = self.norm1(kv)
        attn_out, _ = self.attn(q_norm, kv_norm, kv_norm)
        q = q + attn_out
        q = q + self.mlp(self.norm2(q))
        return q


class SeedAffinityTransformer(nn.Module):
    """Transformer for predicting pairwise seed affinities via cross-attention."""
    def __init__(self, feat_dim=4, dim_coords=2, hidden_dim=64, n_heads=4, n_blocks=2, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(feat_dim, hidden_dim)
        self.norm_in = nn.LayerNorm(hidden_dim)

        self.blocks = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, n_heads, mlp_ratio, dropout)
            for _ in range(n_blocks)
        ])

        self.norm_out = nn.LayerNorm(hidden_dim)

        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + dim_coords, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features, coords):
        """
        features: [K, D] — seed features (embeddings + sigma)
        coords: [K, dim_coords] — actual pixel coordinates
        Returns: M [K, K] — sigmoid affinity matrix
        """
        K = features.shape[0]

        q = self.norm_in(self.proj(features)).unsqueeze(0)  # [1, K, H]
        k = q.clone()

        for block in self.blocks:
            q = block(q, k)
            k = block(k, q)

        q = self.norm_out(q).squeeze(0)  # [K, H]
        k = self.norm_out(k).squeeze(0)  # [K, H]

        # Pairwise features: concatenated representations + relative position
        q_exp = q.unsqueeze(1).expand(-1, K, -1)
        k_exp = k.unsqueeze(0).expand(K, -1, -1)
        rel_pos = (coords.unsqueeze(1) - coords.unsqueeze(0)).float() / 100.0

        pair_feat = torch.cat([q_exp, k_exp, rel_pos], dim=-1)
        logits = self.pair_mlp(pair_feat).squeeze(-1)  # [K, K]
        return torch.sigmoid(logits)


class MyBlock(nn.Sequential):
    def __init__(self, embedding_dim, width):
        super(MyBlock, self).__init__()
        self.fc1 = nn.Conv2d(embedding_dim, width, 1, padding = 0//2)
        self.bn1 = nn.BatchNorm2d(width)
        self.relu1 = nn.ReLU(inplace = True)
        self.fc2 = nn.Conv2d(width, width, 1)
        self.bn2 = nn.BatchNorm2d(width)
        self.relu2 = nn.ReLU(inplace = True)
        self.fc3 = nn.Conv2d(width, 1 , 1)


class ConvProbabilityNet(nn.Module):
    def __init__(self, embedding_dim=4, width = 5, depth = 5):
        super().__init__()
        self.layer1 = MyBlock(embedding_dim + depth,width)
        self.layer2 = MyBlock(embedding_dim ,width)
        self.layer3 = MyBlock(embedding_dim + 2 ,width)
        
        self.positional_embedding_params = (nn.Parameter(torch.rand(depth,3)*10) ).to("cuda")

    

    def forward(self, x, **kwargs):
        # x is C*H*W,E+S+1 (H,W is the window of the crop used here, e.g 100x100, not the image)

        positional_embedding = guide_function(self.positional_embedding_params, width = 100)

        one = self.layer1(torch.cat((x,positional_embedding.expand(x.shape[0],-1,-1,-1)),dim=1))
        two = self.layer2(x)

        output = self.layer3(torch.cat((x,one,two),dim=1))

        return output


class AttentionPixelClassifier(nn.Module):
    def __init__(self, embedding_dim, num_heads=4, head_dim=8, n_sigma=1, rope_theta=10000.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.n_sigma = n_sigma
        dim_coords = embedding_dim - n_sigma

        self.q_proj = nn.Linear(dim_coords, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(embedding_dim, num_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads, 1)

        # Precompute RoPE frequencies (non-learned, moves with model device)
        self.register_buffer('rope_freqs', precompute_rope_freqs_2d(head_dim, theta=rope_theta))

    def forward(self, x, seed_emb=None, window_size=None, **kwargs):
        # x: [C, E+S, H, W] from feature_engineering_attention
        # seed_emb: [C, E] centroid spatial embeddings (dim_coords only)
        C, ES, H, W = x.shape

        # Flatten pixels: [C, H*W, E+S]
        pixel_features = x.permute(0, 2, 3, 1).reshape(C, H * W, ES)

        # Q from seed embedding: [C, 1, dim_coords]
        q = self.q_proj(seed_emb.unsqueeze(1))  # [C, 1, num_heads * head_dim]
        q = q.view(C, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [C, num_heads, 1, head_dim]

        # K from pixel features: [C, H*W, E+S]
        k = self.k_proj(pixel_features)  # [C, H*W, num_heads * head_dim]
        k = k.view(C, H * W, self.num_heads, self.head_dim).transpose(1, 2)  # [C, num_heads, H*W, head_dim]

        # Build relative positions: pixel positions relative to crop center (centroid)
        # The crop is always centered on the centroid, so centroid is at (H//2, W//2)
        ys = torch.arange(H, device=x.device, dtype=x.dtype) - H // 2  # [-H//2, ..., H//2-1]
        xs = torch.arange(W, device=x.device, dtype=x.dtype) - W // 2
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        # pixel_pos: [H*W, 2] — (y, x) relative to centroid
        pixel_pos = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=-1)  # [H*W, 2]
        # Q position is (0, 0) — centroid relative to itself
        q_pos = torch.zeros(1, 2, device=x.device, dtype=x.dtype)  # [1, 2]

        # Apply RoPE rotations to Q and K
        freqs = self.rope_freqs
        q = apply_rope_2d(q, q_pos, freqs)   # [C, num_heads, 1, head_dim]
        k = apply_rope_2d(k, pixel_pos, freqs)  # [C, num_heads, H*W, head_dim]

        # Cross-attention scores (no softmax — raw logits)
        scale = self.head_dim ** 0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # [C, num_heads, 1, H*W]
        scores = scores.squeeze(-2)  # [C, num_heads, H*W]

        # Multi-head -> single logit
        scores = scores.permute(0, 2, 1)  # [C, H*W, num_heads]
        logits = self.out_proj(scores)  # [C, H*W, 1]
        logits = logits.reshape(C * H * W, 1)

        return logits


class RelativeAttentionPixelClassifier(nn.Module):
    def __init__(self, embedding_dim, num_heads=4, head_dim=8, n_sigma=1, max_window=256):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.n_sigma = n_sigma
        dim_coords = embedding_dim - n_sigma

        self.q_proj = nn.Linear(dim_coords, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(embedding_dim, num_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads, 1)
        self.rel_pos_bias = nn.Parameter(torch.zeros(num_heads, max_window, max_window))

    def forward(self, x, seed_emb=None, window_size=None, **kwargs):
        # x: [C, E+S, H, W] from feature_engineering_attention
        # seed_emb: [C, E] centroid spatial embeddings (dim_coords only)
        C, ES, H, W = x.shape

        # Flatten pixels: [C, H*W, E+S]
        pixel_features = x.permute(0, 2, 3, 1).reshape(C, H * W, ES)

        # Q from seed embedding: [C, 1, dim_coords]
        q = self.q_proj(seed_emb.unsqueeze(1))  # [C, 1, num_heads * head_dim]
        q = q.view(C, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [C, num_heads, 1, head_dim]

        # K from pixel features: [C, H*W, E+S]
        k = self.k_proj(pixel_features)  # [C, H*W, num_heads * head_dim]
        k = k.view(C, H * W, self.num_heads, self.head_dim).transpose(1, 2)  # [C, num_heads, H*W, head_dim]

        # Attention scores
        scale = self.head_dim ** 0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # [C, num_heads, 1, H*W]
        scores = scores.squeeze(-2)  # [C, num_heads, H*W]

        # Relative position bias: learned [num_heads, H, W] centered on seed
        bias = self.rel_pos_bias  # [num_heads, max_window, max_window]
        if bias.shape[1] != H or bias.shape[2] != W:
            bias = F.interpolate(bias.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=True).squeeze(0)
        scores = scores + bias.reshape(self.num_heads, H * W).unsqueeze(0)  # broadcast over C

        # Multi-head -> single logit
        scores = scores.permute(0, 2, 1)  # [C, H*W, num_heads]
        logits = self.out_proj(scores)  # [C, H*W, 1]
        logits = logits.reshape(C * H * W, 1)

        return logits


def feature_engineering(x: torch.Tensor, c: torch.Tensor, sigma: torch.Tensor, window_size: int,
                        mesh_grid_flat: torch.Tensor):
    
    
    E = x.shape[0]
    h, w = x.shape[-2:]
    C = c.shape[0]
    S = sigma.shape[0]

    x = torch.cat([x,sigma])[:,mesh_grid_flat[0],mesh_grid_flat[1]]
    
    x = rearrange(x, '(E) (C H W) -> C (E) H W', E = E + S, C = C, H = 2 * window_size, W = 2 * window_size)
    c_shaped = c.view(-1, E, 1, 1)
    x[:,:E] -= c_shaped
    x = rearrange(x, 'C (E) H W-> (C H W) (E)', E = E + S)
    return x



def feature_engineering_slow(x: torch.Tensor, c: torch.Tensor, sigma: torch.Tensor, window_size: int,
                        mesh_grid_flat: torch.Tensor):

    
    E = x.shape[0]
    h, w = x.shape[-2:]
    C = c.shape[0]
    S = sigma.shape[0]

    x_slices = x[:, mesh_grid_flat[0], mesh_grid_flat[1]].reshape(E, C, 2 * window_size, 2 * window_size).permute(1, 0, 2,
                                                                                                                  3)  # C,E,2*window_size,2*window_size
    sigma_slices = sigma[:, mesh_grid_flat[0], mesh_grid_flat[1]].reshape(S, C, 2 * window_size, 2 * window_size).permute(1,
                                                                                                                          0,
                                                                                                                          2,
                                                                                                                          3)  # C,S,2*window_size,2*window_size
    c_shaped = c.reshape(-1, E, 1, 1)

    diff = x_slices - c_shaped

    x = torch.cat([diff, sigma_slices], dim=1)  # C,E+S+1,H,W

    x = x.flatten(2).permute(0, -1, 1)  # C,H*W,E+S+1
    x = x.reshape((x.shape[0] * x.shape[1]), x.shape[2])  # C*H*W,E+S+1

    return x



def feature_engineering_2(x: torch.Tensor, xxyy: torch.Tensor, c: torch.Tensor, sigma: torch.Tensor, window_size: int,
                        mesh_grid_flat: torch.Tensor):
    
    # EXTRA DIFF
    E = x.shape[0]
    h, w = x.shape[-2:]
    C = c.shape[0]
    S = sigma.shape[0]

    x_slices = x[:, mesh_grid_flat[0], mesh_grid_flat[1]].reshape(E, C, 2 * window_size, 2 * window_size).permute(1, 0, 2,3)  # C,E,2*window_size,2*window_size
    sigma_slices = sigma[:, mesh_grid_flat[0], mesh_grid_flat[1]].reshape(S, C, 2 * window_size, 2 * window_size).permute(1,
                                                                                                                          0,
                                                                                                                          2,
                                                                                                                          3)  # C,S,2*window_size,2*window_size
    c_shaped = c.reshape(-1, E, 1, 1)

    norm = torch.sqrt(torch.sum(torch.pow(x_slices - c_shaped, 2) + 1e-6, dim=1, keepdim=True))  # C,1,H,W

    diff = x_slices - c_shaped


    x = torch.cat([diff, sigma_slices, norm], dim=1)  # C,E+S+1,H,W

    x = x.flatten(2).permute(0, -1, 1)  # C,H*W,E+S+1
    x = x.reshape((x.shape[0] * x.shape[1]), x.shape[2])  # C*H*W,E+S+1

    return x

def feature_engineering_3(x: torch.Tensor, xxyy: torch.Tensor, c: torch.Tensor, sigma: torch.Tensor, window_size: int,
                        mesh_grid_flat: torch.Tensor):
    
    # NO SIGMA
    E = x.shape[0]
    h, w = x.shape[-2:]
    C = c.shape[0]
    S = sigma.shape[0]

    x_slices = x[:, mesh_grid_flat[0], mesh_grid_flat[1]].reshape(E, C, 2 * window_size, 2 * window_size).permute(1, 0, 2,
                                                                                                                  3)  # C,E,2*window_size,2*window_size
    sigma_slices = sigma[:, mesh_grid_flat[0], mesh_grid_flat[1]].reshape(S, C, 2 * window_size, 2 * window_size).permute(1,
                                                                                                                          0,
                                                                                                                          2,
                                                                                                                          3)  # C,S,2*window_size,2*window_size
    c_shaped = c.reshape(-1, E, 1, 1)

    diff = x_slices - c_shaped


    x = torch.cat([diff, sigma_slices * 0], dim=1)  # C,E+S+1,H,W

    x = x.flatten(2).permute(0, -1, 1)  # C,H*W,E+S+1
    x = x.reshape((x.shape[0] * x.shape[1]), x.shape[2])  # C*H*W,E+S+1

    return x




def feature_engineering_10(x: torch.Tensor, xxyy: torch.Tensor, c: torch.Tensor, sigma: torch.Tensor, window_size: int,
                        mesh_grid_flat: torch.Tensor):
    
    # CONV
    E = x.shape[0]
    h, w = x.shape[-2:]
    C = c.shape[0]
    S = sigma.shape[0]

    x_slices = x[:, mesh_grid_flat[0], mesh_grid_flat[1]].reshape(E, C, 2 * window_size, 2 * window_size).permute(1, 0, 2,
                                                                                                                  3)  # C,E,2*window_size,2*window_size
    sigma_slices = sigma[:, mesh_grid_flat[0], mesh_grid_flat[1]].reshape(S, C, 2 * window_size, 2 * window_size).permute(1,
                                                                                                                          0,
                                                                                                                          2,
                                                                                                                          3)  # C,S,2*window_size,2*window_size
    c_shaped = c.reshape(-1, E, 1, 1)
    diff = x_slices - c_shaped
    x = torch.cat([diff, sigma_slices], dim=1)  # C,E+S+1,H,W

    return x


def feature_engineering_attention(x: torch.Tensor, c: torch.Tensor, sigma: torch.Tensor, window_size: int,
                                  mesh_grid_flat: torch.Tensor):
    """Extract crop embeddings with centroid subtraction, returning 4D tensor
    [C, E+S, H, W] for use with AttentionPixelClassifier.
    RoPE in the classifier provides additional relative position encoding."""
    E = x.shape[0]
    C = c.shape[0]
    S = sigma.shape[0]
    x = torch.cat([x, sigma])[:, mesh_grid_flat[0], mesh_grid_flat[1]]
    x = rearrange(x, '(E) (C H W) -> C (E) H W',
                  E=E + S, C=C, H=2 * window_size, W=2 * window_size)
    # Subtract centroid embeddings from spatial channels (not sigma)
    c_shaped = c.view(-1, E, 1, 1)
    x[:, :E] -= c_shaped
    return x


def compute_seed_merge_matrix(seed_embs, seed_sigmas, affinity_net, coords):
    """
    seed_embs: [K, E] — seed embedding vectors
    seed_sigmas: [K, S] — sigma values at seed locations
    affinity_net: SeedAffinityTransformer
    coords: [K, dim_coords] — actual pixel coordinates of seeds
    Returns: M [K, K] — sigmoid affinity matrix
    """
    features = torch.cat([seed_embs, seed_sigmas], dim=-1)  # [K, E+S]
    return affinity_net(features, coords)


def _gate_by_center_logit(dist, centroids, h, w, window_size):
    """Gate per-seed instance logits by the MLP's own prediction at the seed center.

    Computes logit(sigmoid(a) * sigmoid(b)) = a + b - log(1 + exp(a) + exp(b))
    entirely in logit space using numerically stable ops (no sigmoid/logit roundtrip).
    This suppresses all predictions from seeds whose center logit is low (background),
    while leaving confident seeds unchanged.

    dist: [K, 1, ws, ws] — per-seed logits from compute_crops
    centroids: [K, 2] — (y, x) pixel coordinates of seeds
    h, w: image dimensions
    window_size: crop window size
    Returns: gated dist [K, 1, ws, ws]
    """
    K = dist.shape[0]
    ws = window_size

    # Find each centroid's position within its crop (replicates centre_crop clamping)
    clamped_y = centroids[:, 0].clamp(min=ws // 2, max=h - ws // 2)
    clamped_x = centroids[:, 1].clamp(min=ws // 2, max=w - ws // 2)
    cy = (centroids[:, 0] - (clamped_y - ws // 2)).long()
    cx = (centroids[:, 1] - (clamped_x - ws // 2)).long()

    center_logit = dist[torch.arange(K, device=dist.device), 0, cy, cx]  # [K]
    b = center_logit.view(K, 1, 1, 1)

    # logit(sigmoid(a) * sigmoid(b)) = a + b - log(1 + exp(a) + exp(b))
    # = a + b - softplus(logaddexp(a, b))
    return dist + b - F.softplus(torch.logaddexp(dist, b))


def apply_seed_merging(dist, coords, M, h, w, window_size):
    """
    Apply M to predictions in global pixel space (differentiable).
    Performs a per-pixel weighted average in logit space, only over seeds
    that actually cover each pixel (seeds without predictions are excluded,
    not treated as predicting background).
    M is symmetrized and row-normalized per-pixel.
    dist: [K, 1, ws, ws] — per-seed logits
    coords: [3, K*ws*ws] — coords[0]=seed_idx, coords[1]=y, coords[2]=x
    M: [K, K] — seed affinity matrix (sigmoid values in [0,1])
    h, w: image dimensions
    window_size: crop window size
    Returns: merged dist [K, 1, ws, ws]
    """
    K = dist.shape[0]
    seed_idx = coords[0].long()
    pixel_idx = (coords[1] * w + coords[2]).long()

    # Symmetrize M and force diagonal to 1
    M = (M + M.T) / 2
    M.fill_diagonal_(1.0)

    # Dense logit tensor + coverage mask
    logits = torch.zeros((K, h * w), device=dist.device, dtype=dist.dtype)
    has_pred = torch.zeros((K, h * w), device=dist.device, dtype=dist.dtype)
    logits[seed_idx, pixel_idx] = dist.flatten()
    has_pred[seed_idx, pixel_idx] = 1.0

    # Weighted average in logit space, only over seeds that cover each pixel
    merged = (M @ (logits * has_pred)) / (M @ has_pred).clamp(min=1e-6)

    merged_dist = merged[seed_idx, pixel_idx]
    return merged_dist.view(K, 1, window_size, window_size)



def feature_engineering_generator(feature_engineering_function):

    if feature_engineering_function == "0" or feature_engineering_function == "7":
        return feature_engineering, 2
    elif feature_engineering_function == "2":
        return feature_engineering_2, 3
    elif feature_engineering_function == "3":
        return feature_engineering_3, 2
    elif feature_engineering_function == "10":
        return feature_engineering_10, 2
    elif feature_engineering_function == "attention":
        return feature_engineering_attention, 2
    elif feature_engineering_function == "attention_relative":
        return feature_engineering_attention, 2

    else:
        raise NotImplementedError("Feature engineering function",feature_engineering_function,"is not implemented")

class InstanSeg(nn.Module):

    def __init__(self,
                 n_sigma: int = 1,
                 instance_weight: float = 1.5,
                 device: str = 'cuda',
                 instance_loss_fn_str: str = "lovasz_hinge",
                 seed_loss_fn = "ce",
                 cells_and_nuclei: bool = False,
                 window_size = 256,
                 feature_engineering_function = "0",
                 bg_weight = None,
                 dim_coords = 2,
                 dim_seeds = 1,
                 mask_loss_fn = None,
                 seed_merging = False,
                 uncertainty_weighting = False,
                 batched_instance_loss = True,):

        super().__init__()
        self.n_sigma = n_sigma
        self.instance_weight = instance_weight
        self.uncertainty_weighting = uncertainty_weighting
        self.batched_instance_loss = batched_instance_loss
        if uncertainty_weighting:
            self.log_var_seed = nn.Parameter(torch.zeros(1))
            self.log_var_inst = nn.Parameter(torch.zeros(1))
        self.device = device
        self.dim_coords = dim_coords

        self.dim_seeds = dim_seeds

        self.dim_out = self.dim_coords + self.n_sigma + self.dim_seeds
        self.parameters_have_been_updated = False

        if cells_and_nuclei:
            self.dim_out = self.dim_out * 2
        self.cells_and_nuclei = cells_and_nuclei
        self.window_size = window_size
        self.num_instance_cap = 50
        self.bg_weight = bg_weight
        self.seed_merging = seed_merging

        self.feature_engineering, self.feature_engineering_width = feature_engineering_generator(feature_engineering_function)
        self.feature_engineering_function = feature_engineering_function
        self.coord_mode = "rope" if feature_engineering_function == "attention" else "linear"  # attention_relative uses linear

        self.update_instance_loss(instance_loss_fn_str)
        self.update_seed_loss(seed_loss_fn)
        self.update_mask_loss(mask_loss_fn)

    def update_instance_loss(self, instance_loss_fn_str):

        if instance_loss_fn_str == "lovasz_hinge":
            if self.batched_instance_loss:
                from instanseg.utils.loss.lovasz_losses import lovasz_hinge_batched
                def instance_loss_fn(pred, gt, **kwargs):
                    return lovasz_hinge_batched(pred.squeeze(1), gt)
            else:
                from instanseg.utils.loss.lovasz_losses import lovasz_hinge
                def instance_loss_fn(pred, gt, **kwargs):
                    return lovasz_hinge(pred.squeeze(1), gt, per_image=True)

        elif instance_loss_fn_str == "ce":
            self.instance_loss_fn = torch.nn.BCEWithLogitsLoss()
            return

        elif instance_loss_fn_str == "dicefocal_loss":
            from monai.losses import DiceFocalLoss
            instance_loss_fn_ = DiceFocalLoss(sigmoid=True)
            def instance_loss_fn(pred, gt, **kwargs):
                l = instance_loss_fn_(pred[None,:,0], gt.unsqueeze(0)) * 1.5
                return l
        elif instance_loss_fn_str == "dice_loss":
            from monai.losses import DiceLoss
            instance_loss_fn_ = DiceLoss(sigmoid=True)
            def instance_loss_fn(pred, gt, **kwargs):
                l = instance_loss_fn_(pred[None,:,0], gt.unsqueeze(0)) * 1.5
                return l

        elif instance_loss_fn_str == "scnp_dice_loss":
            # Spatially-Coherent Neighborhood Pooling — https://arxiv.org/pdf/2603.18671
            # Replaces each logit with the weakest same-class logit in a 3x3
            # window (min over FG, max over BG) before BCE+Dice.
            from monai.losses import DiceCELoss
            base_loss = DiceCELoss(sigmoid=True)
            kernel_size = 3
            padding = kernel_size // 2
            LARGE = 1.0e4
            def instance_loss_fn(pred, gt, **kwargs):
                logits = pred.squeeze(1)                       # (K, H, W)
                Y = (gt > 0.5).float()                         # (K, H, W)
                logits_4d = logits.unsqueeze(1)                # (K, 1, H, W)
                Y4 = Y.unsqueeze(1)
                t1 = -F.max_pool2d(-(logits_4d * Y4 + LARGE * (1 - Y4)),
                                   kernel_size=kernel_size, stride=1, padding=padding)
                t2 = F.max_pool2d(logits_4d * (1 - Y4) - LARGE * Y4,
                                  kernel_size=kernel_size, stride=1, padding=padding)
                scnp_logits = (t1 * Y4 + t2 * (1 - Y4)).squeeze(1)  # (K, H, W)
                l = base_loss(scnp_logits.unsqueeze(0), Y.unsqueeze(0)) * 1.5
                return l

        elif instance_loss_fn_str == "general_dice_loss":
            from monai.losses import GeneralizedDiceLoss
            def instance_loss_fn(pred, gt):
                return GeneralizedDiceLoss(sigmoid=True)(pred, gt.unsqueeze(1))
        else:
            raise NotImplementedError(f"Instance loss function '{instance_loss_fn_str}' is not implemented")
        self.instance_loss_fn = instance_loss_fn

    def update_seed_loss(self, seed_loss_fn):
        if seed_loss_fn is None or seed_loss_fn == "none":
            self.seed_loss = None
            return

        if seed_loss_fn in ["ce"]:
            binary_loss = torch.nn.BCEWithLogitsLoss(reduction='none')

            def seed_loss(x, y, mask=None):
                if mask is not None:
                    mask = mask.float()
                    loss = binary_loss(x, (y > 0).float())
                    masked_loss = loss * mask
                    return masked_loss.sum() / mask.sum().clamp(min=1)
                else:
                    return binary_loss(x, (y > 0).float()).mean()

            self.seed_loss = seed_loss

        elif seed_loss_fn in ["l1_distance", "l2_distance"]:
            from instanseg.utils.pytorch_utils import instance_wise_edt

            if seed_loss_fn == "l1_distance":
                distance_loss = torch.nn.L1Loss(reduction='none')
            elif seed_loss_fn == "l2_distance":
                distance_loss = torch.nn.MSELoss(reduction='none')

            def seed_loss(x, y, mask=None):
                edt = (instance_wise_edt(y.float(), edt_type='edt') - 0.5) * 15
                loss = distance_loss((x), (edt[None]))

                if self.bg_weight is not None:
                    weights = torch.where(edt < 0, self.bg_weight, 1.0)
                    loss = loss * weights

                if mask is not None:
                    mask = mask.float()
                    masked_loss = loss * mask
                    return masked_loss.sum() / mask.sum().clamp(min=1)
                else:
                    return loss.mean()

            self.seed_loss = seed_loss

        elif seed_loss_fn == "l1_poisson":
            from instanseg.utils.pytorch_utils import instance_wise_poisson

            distance_loss = torch.nn.L1Loss(reduction='none')

            def seed_loss(x, y, mask=None):
                poisson = (instance_wise_poisson(y.float()) - 0.5) * 15
                loss = distance_loss(x, poisson[None])
                if self.bg_weight is not None:
                    weights = torch.where(poisson < 0, self.bg_weight, 1.0)
                    loss = loss * weights

                if mask is not None:
                    mask = mask.float()
                    masked_loss = loss * mask
                    return masked_loss.sum() / mask.sum().clamp(min=1)
                else:
                    return loss.mean()

            self.seed_loss = seed_loss

        else:
            raise NotImplementedError(f"Seed loss function '{seed_loss_fn}' is not implemented")

    def update_mask_loss(self, mask_loss_fn):
        self.mask_loss_fn_str = mask_loss_fn
        if mask_loss_fn is None:
            self.mask_loss = None
        elif mask_loss_fn == "ce":
            bce = torch.nn.BCEWithLogitsLoss(reduction='none')
            def mask_loss(x, y, mask=None, bg_weight=None):
                loss = bce(x, (y > 0).float())
                if bg_weight is not None:
                    loss = loss * torch.where(y == 0, bg_weight, 1.0)
                if mask is not None:
                    mask = mask.float()
                    return (loss * mask).sum() / (mask.sum() + 1e-4)
                return loss.mean()
            self.mask_loss = mask_loss
        elif mask_loss_fn == "dice":
            from monai.losses import DiceLoss
            dice = DiceLoss(sigmoid=True)
            def mask_loss(x, y, mask=None, bg_weight=None):
                return dice(x, (y > 0).float())
            self.mask_loss = mask_loss
        else:
            raise NotImplementedError(f"Mask loss function '{mask_loss_fn}' is not implemented")

    def initialize_pixel_classifier(self, model, MLP_width = 10, MLP_input_dim = None):

        if has_pixel_classifier_model(model):
            try:
                self.pixel_classifier = model.pixel_classifier
            except:
                self.pixel_classifier = model.model.pixel_classifier  # This happens when there is an adaptornet

            # Load existing seed affinity net if present, or create a new one
            if self.seed_merging:
                try:
                    self.seed_affinity_net = model.seed_affinity_net
                except AttributeError:
                    try:
                        self.seed_affinity_net = model.model.seed_affinity_net
                    except AttributeError:
                        affinity_feat_dim = self.dim_coords + self.n_sigma
                        model.seed_affinity_net = SeedAffinityTransformer(
                            feat_dim=affinity_feat_dim,
                            dim_coords=self.dim_coords,
                        )
                        self.seed_affinity_net = model.seed_affinity_net.to(self.device)

            if self.uncertainty_weighting:
                model.log_var_seed = self.log_var_seed
                model.log_var_inst = self.log_var_inst

            return model
        else:
            if MLP_input_dim is None:
                MLP_input_dim = self.feature_engineering_width + self.n_sigma -2 + self.dim_coords
            if self.feature_engineering_function == "attention":
                model.pixel_classifier = AttentionPixelClassifier(
                    embedding_dim=MLP_input_dim,
                    num_heads=4,
                    head_dim=8,
                    n_sigma=self.n_sigma,
                )
            elif self.feature_engineering_function == "attention_relative":
                model.pixel_classifier = RelativeAttentionPixelClassifier(
                    embedding_dim=MLP_input_dim,
                    num_heads=4,
                    head_dim=8,
                    n_sigma=self.n_sigma,
                    max_window=self.window_size,
                )
            elif self.feature_engineering_function != "10":
                model.pixel_classifier = ProbabilityNet( MLP_input_dim, width = MLP_width)
            else:
                model.pixel_classifier = ConvProbabilityNet( MLP_input_dim, width = MLP_width)
            self.pixel_classifier = model.pixel_classifier.to(self.device)

            # Create separate seed affinity network if seed merging is enabled
            if self.seed_merging:
                affinity_feat_dim = self.dim_coords + self.n_sigma
                model.seed_affinity_net = SeedAffinityTransformer(
                    feat_dim=affinity_feat_dim,
                    dim_coords=self.dim_coords,
                )
                self.seed_affinity_net = model.seed_affinity_net.to(self.device)

            # Register uncertainty parameters on model so optimizer picks them up
            if self.uncertainty_weighting:
                model.log_var_seed = self.log_var_seed
                model.log_var_inst = self.log_var_inst

            return model


    def forward(self, prediction: torch.Tensor, instances: torch.Tensor, w_inst: float = 1.5, w_seed: float = 1.0):

        w_inst = self.instance_weight

        batch_size, height, width = prediction.size(
            0), prediction.size(2), prediction.size(3)

        xxyy = generate_coordinate_map(mode = self.coord_mode, spatial_dim = self.dim_coords, height = height, width = width, device = prediction.device)

        seed_loss_sum = 0
        total_seed_loss = 0
        instance_loss_sum = 0
        all_dists = []
        all_crops = []

        if self.cells_and_nuclei:
            dim_out = int(self.dim_out / 2)
        else:
            dim_out = self.dim_out

        for mask_channel in range(0, instances.shape[1]):

            if mask_channel == 0:
                prediction_b = prediction[:, 0: dim_out, :, :]
            else:
                prediction_b = prediction[:, dim_out:, :, :]

            instances_batch = instances

            spatial_emb_batch = (torch.sigmoid((prediction_b[:, 0: self.dim_coords]))-0.5) * 8 + xxyy

            sigma_batch = prediction_b[:, self.dim_coords: self.dim_coords + self.n_sigma]  # n_sigma x h x w
            seed_map_batch = prediction_b[:, - self.dim_seeds:]  # 1 x h x w


            for b in range(0, batch_size):

                spatial_emb = spatial_emb_batch[b]
                sigma = sigma_batch[b]
                seed_map = seed_map_batch[b]

                seed_loss = 0

                instance = instances_batch[b, mask_channel].unsqueeze(0)  # 1 x h x w

                if (instance < 0).all(): #-1 means not annotated
                    continue

                elif instance.min() < 0: #label is sparse
                    mask = instance >=0
                    instance[instance < 0] = 0
                else:
                    mask = None


                if self.seed_loss is not None:
                    if self.mask_loss is not None and self.dim_seeds == 2:
                        fg_mask = (instance > 0)
                        if mask is not None:
                            fg_mask = fg_mask & mask
                        seed_loss_tmp = self.seed_loss(seed_map[0:1], instance, mask=fg_mask)
                        seed_loss_tmp += self.mask_loss(seed_map[1:2], instance, mask=mask, bg_weight=self.bg_weight)
                    else:
                        seed_loss_tmp = self.seed_loss(seed_map, instance, mask=mask)

                    seed_loss += seed_loss_tmp

                if w_inst == 0:
                    seed_loss_sum += w_seed * seed_loss
                    total_seed_loss += seed_loss
                    continue

                if instance.min() > 0:
                    seed_loss_sum += w_seed * seed_loss
                    total_seed_loss += seed_loss
                    continue

                instance_ids = instance.unique()
                instance_ids = instance_ids[instance_ids != 0]

                if len(instance_ids) > 0:

                    instance = torch_fastremap(instance)

                    onehot_labels = torch_onehot(instance).squeeze(0)  # C x h x w

                    self.min_gt_instanseg_area = 10
                    if self.min_gt_instanseg_area is not None:
                        onehot_labels = onehot_labels[onehot_labels.sum((1,2)) > self.min_gt_instanseg_area]
                        if onehot_labels.shape[0] == 0:
                            seed_loss_sum += w_seed * seed_loss
                            total_seed_loss += seed_loss
                            continue

                    if self.num_instance_cap is not None: #This is to cap the number of objects to avoid OOM errors.
                         if self.num_instance_cap < onehot_labels.shape[0]:
                            idx = torch.randperm(onehot_labels.shape[0])[:self.num_instance_cap]
                            onehot_labels = onehot_labels[idx]


                    seed_map_tmp = torch.sigmoid(seed_map[0]) #note seed_map may have 2 channels, we keep the first one.


                    centroids = torch_peak_local_max(seed_map_tmp.squeeze() * onehot_labels.sum(0), neighbourhood_size = 3, minimum_value = 0.5).T

                    centres = spatial_emb[:,centroids[0],centroids[1]].detach().T

                    idx = torch.randperm(centroids.shape[1])[:self.num_instance_cap]

                    centres = centres[idx]
                    centroids = centroids[:,idx]

                    instance_labels = onehot_labels[:,centroids[0],centroids[1]].float().argmax(0)
                    onehot_labels = onehot_labels[instance_labels]
                    centroids = centroids.T


                    if len(centroids) == 0:
                        seed_loss_sum += w_seed * seed_loss
                        total_seed_loss += seed_loss
                        continue

                    window_size = min(self.window_size, height, width)

                    dist, coords = compute_crops(spatial_emb,
                                                 centres,
                                                 sigma,
                                                 centroids,
                                                 feature_engineering = self.feature_engineering,
                                                 pixel_classifier=self.pixel_classifier,
                                                 window_size = window_size)

                    # Seed merging: pool predictions across seeds via M
                    if self.seed_merging and hasattr(self, 'seed_affinity_net'):
                        sigma_at_seeds = sigma[:, centroids[:, 0], centroids[:, 1]].T  # [K, S]
                        M = compute_seed_merge_matrix(centres, sigma_at_seeds, self.seed_affinity_net, centroids.float())
                        dist = apply_seed_merging(dist, coords, M, height, width, window_size)

                    # No-seed-loss mode: gate predictions by center logit (AND in probability space)
                    if self.seed_loss is None:
                        dist = _gate_by_center_logit(dist, centroids, height, width, window_size)

                    crop = onehot_labels.squeeze(1)[coords[0], coords[1], coords[2]].reshape(-1,window_size, window_size)
                    if self.batched_instance_loss:
                        all_dists.append(dist)
                        all_crops.append(crop.float())
                    else:
                        image_dists = [dist]
                        image_crops = [crop.float()]

                    # Grid-ensured maxima: fill empty grid cells with additional seeds
                    if self.seed_merging and (grid_maxima := ensure_grid_maxima(seed_map_tmp.squeeze(), centroids, grid_size=16)).shape[0] > 0:
                        grid_centres = spatial_emb[:, grid_maxima[:, 0], grid_maxima[:, 1]].detach().T

                        dist_grid, coords_grid = compute_crops(spatial_emb,
                                                                grid_centres,
                                                                sigma,
                                                                grid_maxima,
                                                                feature_engineering=self.feature_engineering,
                                                                pixel_classifier=self.pixel_classifier,
                                                                window_size=window_size)

                        if self.seed_loss is None:
                            dist_grid = _gate_by_center_logit(dist_grid, grid_maxima, height, width, window_size)

                        # Build target: instance mask for FG seeds, all zeros for BG seeds
                        labels_at_grid = instance[0, grid_maxima[:, 0], grid_maxima[:, 1]]  # [K_grid]
                        grid_target = (instance[0:1] == labels_at_grid.view(-1, 1, 1)).float()  # [K_grid, H, W]
                        grid_target[labels_at_grid == 0] = 0  # background seeds → predict nothing

                        crop_grid = grid_target[coords_grid[0].long(), coords_grid[1].long(), coords_grid[2].long()].reshape(-1, window_size, window_size)
                        if self.batched_instance_loss:
                            all_dists.append(dist_grid)
                            all_crops.append(crop_grid.float())
                        else:
                            image_dists.append(dist_grid)
                            image_crops.append(crop_grid.float())

                    # Unbatched: compute instance loss per image
                    if not self.batched_instance_loss:
                        img_d = torch.cat(image_dists, dim=0)
                        img_c = torch.cat(image_crops, dim=0)
                        instance_loss_sum += self.instance_loss_fn(img_d, img_c)

                seed_loss_sum += w_seed * seed_loss
                total_seed_loss += seed_loss

        # Average seed loss over batch
        avg_seed_loss = seed_loss_sum / (b + 1)

        # Compute instance loss
        if self.batched_instance_loss:
            if all_dists:
                all_dists = torch.cat(all_dists, dim=0)
                all_crops = torch.cat(all_crops, dim=0)
                total_instance_loss = self.instance_loss_fn(all_dists, all_crops)
            else:
                total_instance_loss = 0
        else:
            total_instance_loss = instance_loss_sum / (b + 1)

        # Combine seed and instance losses
        if self.uncertainty_weighting:
            loss = 0
            if isinstance(avg_seed_loss, torch.Tensor):
                loss = loss + torch.exp(-self.log_var_seed) * avg_seed_loss + self.log_var_seed
            if isinstance(total_instance_loss, torch.Tensor):
                loss = loss + torch.exp(-self.log_var_inst) * total_instance_loss + self.log_var_inst
        else:
            loss = avg_seed_loss + w_inst * total_instance_loss

        if self.cells_and_nuclei:
            loss = loss / 2

        if type(loss) != torch.Tensor:
            loss = spatial_emb * 0

        # Store component losses for logging
        self.last_seed_loss = float(total_seed_loss) / (b + 1)
        self.last_instance_loss = float(total_instance_loss)

        return loss
    
    
    def reset_uncertainty_weights(self):
        if self.uncertainty_weighting:
            self.log_var_seed.data.zero_()
            self.log_var_inst.data.zero_()

    def update_hyperparameters(self,params):
        self.parameters_have_been_updated = True
        self.params = params


    #@timer
    def postprocessing(self, prediction: Union[torch.Tensor, np.ndarray],
                        mask_threshold: float = 0.53,
                        peak_distance: int = 4,
                        seed_threshold: float = 0.5,
                        overlap_threshold: float = 0.5,
                        mean_threshold: float = -10000,
                        fg_threshold: float = 0.5, #not used in default instanseg
                        window_size: int = None,
                        min_size = 10,
                        device=None,
                        classifier=None,
                        cleanup_fragments: bool = False,
                        max_seeds: int = 2000,
                        return_intermediate_objects: bool = False,
                        precomputed_crops: torch.Tensor = None,
                        precomputed_seeds: torch.Tensor = None,
                        img=None,
                        overlap_metric: str = "iou"):
            

        if window_size is None:
            window_size = self.window_size
        if device is None:
            device = self.device
        if classifier is None:
            classifier = self.pixel_classifier

        if self.parameters_have_been_updated:
            for key in self.params:
                setattr(self, key, self.params[key])
            # mask_threshold = self.params['mask_threshold']
            # peak_distance = self.params['peak_distance']
            # seed_threshold = self.params['seed_threshold']
            # overlap_threshold = self.params['overlap_threshold']
            # if "min_size" in self.params:
            #     min_size = self.params['min_size']
            # if "mean_threshold" in self.params:
            #     mean_threshold = self.params['mean_threshold']


        if isinstance(prediction, np.ndarray):
            prediction = torch.tensor(prediction, device=device)

        if self.cells_and_nuclei:
            iterations = 2
            dim_out = int(self.dim_out / 2)
        else:
            iterations = 1
            dim_out = self.dim_out

        labels = []

        for i in range(iterations):
            M = None

            if precomputed_crops is None:

                if i == 0:
                    prediction_i = prediction[0: dim_out, :, :]
                else:
                    prediction_i = prediction[dim_out:, :, :]

                height, width = prediction_i.size(1), prediction_i.size(2)

                xxyy = generate_coordinate_map(mode = self.coord_mode, spatial_dim = self.dim_coords, height = height, width = width, device = device)

                fields = (torch.sigmoid(prediction_i[0:self.dim_coords])-0.5) * 8

                sigma = prediction_i[self.dim_coords:self.dim_coords + self.n_sigma]

                if self.dim_seeds == 1:
                    mask_map = ((prediction_i[-1]) / 15) + 0.5
                    binary_map = mask_map
                else:
                    binary_map = torch.sigmoid(prediction_i[-1])
                    mask_map = (prediction_i[-2] / 15).clone() + 0.5

                    mask_map[~(binary_map > fg_threshold)] = 0  # Set seed_map to 0 where binary_map is False


                if (mask_map > mask_threshold).max() == 0:  # no foreground pixels
                    label = torch.zeros(mask_map.shape, dtype=int, device=mask_map.device).squeeze()
                    labels.append(label)
                    continue

                if precomputed_seeds is None:
                    local_centroids_idx = torch_peak_local_max(mask_map, neighbourhood_size=int(peak_distance), minimum_value=seed_threshold)
                else:
                    local_centroids_idx = precomputed_seeds

            
                fields = fields + xxyy

                fields_at_centroids = fields[:, local_centroids_idx[:, 0], local_centroids_idx[:, 1]]

                if local_centroids_idx.shape[0] > max_seeds:
                    print("Too many seeds, skipping", local_centroids_idx.shape[0])
                    label = torch.zeros(mask_map.shape, dtype=int, device=mask_map.device).squeeze()
                    labels.append(label)
                    continue
                
                C = fields_at_centroids.shape[1]

                h, w = mask_map.shape[-2:]
                window_size = min(window_size, h, w)
                window_size = window_size - window_size % 2

                if C == 0:
                    label = torch.zeros(mask_map.shape, dtype=int, device=mask_map.device).squeeze()
                    labels.append(label)
                    continue

                # Seed merging: compute affinity matrix for label merging only
                if self.seed_merging and hasattr(self, 'seed_affinity_net'):
                    sigma_at_seeds = sigma[:, local_centroids_idx[:, 0], local_centroids_idx[:, 1]].T  # [K, S]
                    M = compute_seed_merge_matrix(fields_at_centroids.T, sigma_at_seeds, self.seed_affinity_net, local_centroids_idx.float())

                crops, coords = compute_crops(fields,
                                                fields_at_centroids.T,
                                                sigma,
                                                local_centroids_idx.int(),
                                                feature_engineering = self.feature_engineering,
                                                pixel_classifier=classifier,
                                                mask_threshold=mask_threshold,
                                                cleanup_fragments= cleanup_fragments,
                                                window_size=window_size) # about 65% of the time

                # Seed merging: apply soft pooling to crops (matching training)
                if M is not None:
                    crops = apply_seed_merging(crops, coords, M, h, w, window_size)

                # No-seed-loss mode: gate predictions by center logit (matching training)
                if self.seed_loss is None:
                    crops = _gate_by_center_logit(crops, local_centroids_idx, h, w, window_size)

                coords = coords[1:] # The first channel are just channel indices, not required here.


                if return_intermediate_objects:
                    return crops, coords, mask_map

                C = crops.shape[0]
                if C == 0:
                    label = torch.zeros(mask_map.shape, dtype=int, device=mask_map.device).squeeze()
                    labels.append(label)
                    continue

            else:
                crops,coords,mask_map = precomputed_crops
                C = crops.shape[0]


            h, w = mask_map.shape[-2:]

            crops = torch.sigmoid(crops) 
      
            label = merge_sparse_predictions(crops,
                                            coords,
                                            binary_map,
                                            size=(C,h, w),
                                            mask_threshold=mask_threshold,
                                            window_size=window_size,
                                            min_size=min_size,
                                            overlap_threshold=overlap_threshold,
                                            mean_threshold=mean_threshold,
                                            seed_affinity=M,
                                            overlap_metric=overlap_metric).int() #about 30% of the time
            
            labels.append(label.squeeze())


        if len(labels) == 1:
            return labels[0][None]  # 1,H,W
        else:
            return torch.stack(labels)  # 2,H,W
        

    def TTA_postprocessing(self, img, model, transforms,
                        mask_threshold: float = 0.53,
                        peak_distance: int = 5,
                        seed_threshold: float = 0.8,
                        overlap_threshold: float = 0.3,
                        mean_threshold: float = 0.1,
                        window_size: int =64,
                        min_size = 10,
                       device=None,
                       classifier=None,
                       cleanup_fragments: bool = False,
                       reduction = "mean",
                       max_seeds: int = 2000,):

        
        cells_and_nuclei = self.cells_and_nuclei
        if self.cells_and_nuclei:
            iterations = 2
            assert self.dim_out % 2 == 0,  print("The model should an even number of output channels for cells and nuclei.")
            dim_out = int(self.dim_out / 2)
        else:
            iterations = 1
            dim_out = self.dim_out

        out_labels = []

        transforms = [t for t in transforms] + [IdentityTransform()]
        
        for i in range(iterations):

            all_masks_list = []
            all_predictions = []

            self.cells_and_nuclei = False

            for t in transforms:
                with torch.amp.autocast("cuda"):
                    augmented_image = t.augment_image(img)
                    augmented_image, pad = _instanseg_padding(augmented_image, extra_pad= 0, min_dim = 32)
                    prediction = model(augmented_image)[:,i * dim_out:(i+1) * dim_out]
                    prediction = _recover_padding(prediction, pad)
                    mask_map = prediction[:,-1][None] 
                    mask_map = t.deaugment_mask(mask_map)
                    all_masks_list.append(mask_map.cpu())
                    all_predictions.append(prediction.cpu())

            if reduction == "local_max":

                local_maxima_maps = [torch_peak_local_max(mask.squeeze().float().to(device), int(peak_distance),seed_threshold, return_map = True) for mask in all_masks_list]
                local_maxima_map = torch_peak_local_max(torch.stack(local_maxima_maps).max(0)[0].squeeze(),int(peak_distance),seed_threshold, return_map = True)
                all_masks = torch.mean(torch.stack(all_masks_list),dim=0).float().to(device)

            elif reduction in ["mean", "median"]:
                if reduction == "mean":
                    all_masks = torch.mean(torch.stack(all_masks_list),dim=0).float().to(device)
                elif reduction == "median":
                    all_masks = torch.median(torch.stack(all_masks_list),dim=0)[0].to(device)

                local_maxima_map = torch_peak_local_max(all_masks.squeeze(), neighbourhood_size=int(peak_distance), minimum_value=seed_threshold, return_map = True)
            
            local_maxima_map = (local_maxima_map > 0).float()
            centroids = torch.stack(torch.where(local_maxima_map.squeeze())).T


            if len(centroids) == 0:
                out_labels.append(torch.zeros((1,*all_masks.shape[-2:]), dtype=int, device=device))
                continue

            local_maxima_map[...,centroids[:,0],centroids[:,1]] = torch.arange(1,centroids.shape[0]+1,device = local_maxima_map.device).float()

            all_crops = []

            for (t, prediction) in (zip(transforms, all_predictions)):
                prediction_tmp = prediction.clone().float().to(device)
                prediction_tmp[:,-1] = t.augment_image(all_masks)
                prediction_tmp = prediction_tmp.squeeze(0)

                local_maxima_map_tmp = t.augment_image( local_maxima_map )
                centroids = torch.stack(torch.where(local_maxima_map_tmp.squeeze())).T
                values = local_maxima_map_tmp[...,centroids[:,0],centroids[:,1]]
                centroids = centroids[values.sort()[1]][0,0]

                out = self.postprocessing(prediction_tmp, mask_threshold, peak_distance, seed_threshold, overlap_threshold, mean_threshold, window_size, min_size, device, classifier, 
                                                            cleanup_fragments, max_seeds, return_intermediate_objects = True, precomputed_seeds = centroids)
                
                if len(out)==3:
                    crops, coords, mask_map = out
                else:
                    pdb.set_trace()

                crops = t.deaugment_mask(crops)
                all_crops.append(crops.cpu())

         #   show_images(torch.cat([*torch.cat(all_crops,dim = 3)[:50]],dim = 1),colorbar= False)

            all_crops = torch.median(torch.stack(all_crops).float(),dim=0)[0].to(device)
         #   all_crops = torch.mean(torch.stack(all_crops).float(),dim=0).to(device)
         #   all_crops = torch.max(torch.stack(all_crops).float(),dim=0)[0].to(device)

            labels = self.postprocessing(prediction, mask_threshold, peak_distance, seed_threshold, overlap_threshold, mean_threshold, window_size, min_size, device, classifier,
                                        cleanup_fragments, max_seeds, precomputed_crops = (all_crops, coords, mask_map))

            out_labels.append(labels)
        self.cells_and_nuclei = cells_and_nuclei

        labels = torch.stack(out_labels, dim = 1).squeeze(0)
        #show_images(labels)
        return labels


class IdentityTransform:
    def augment_image(self, img):
        return img
    def deaugment_mask(self, mask):
        return mask

        
from instanseg.utils.biological_utils import resolve_cell_and_nucleus_boundaries
from typing import Dict, Optional
class InstanSeg_Torchscript(nn.Module):
    def __init__(self, model, 
                 cells_and_nuclei: bool = False,
                 pixel_size : float = 0, 
                 n_sigma: int = 2, 
                 dim_coords:int = 2, 
                 dim_seeds: int = 1,
                 backbone_dim_in: int = 3,  
                 feature_engineering_function:str  = "0",
                 params = None):
        super(InstanSeg_Torchscript, self).__init__()

        model.eval()

        use_mixed_precision = True

        with torch.amp.autocast("cuda", enabled=use_mixed_precision):
            with torch.no_grad():
                self.fcn = torch.jit.trace(model, torch.rand(1, backbone_dim_in, 256, 256))

        #from instanseg.utils.models.CellposeSam import SAM_UNet_inference
        #self.fcn = SAM_UNet_inference(self.fcn)

        try:
            self.pixel_classifier = model.pixel_classifier
        except:
            self.pixel_classifier = model.model.pixel_classifier

        # Load seed affinity net if present
        try:
            self.seed_affinity_net = model.seed_affinity_net
        except AttributeError:
            try:
                self.seed_affinity_net = model.model.seed_affinity_net
            except AttributeError:
                pass

        self.cells_and_nuclei = cells_and_nuclei
        self.pixel_size = pixel_size
        self.dim_coords = dim_coords
        self.dim_seeds = dim_seeds
        self.n_sigma = n_sigma
        self.feature_engineering, self.feature_engineering_width = feature_engineering_generator(feature_engineering_function)
        self.params = params or {}
        self.index_dtype = torch.long #torch.int

        self.default_target_segmentation = self.params.get('target_segmentation', torch.tensor([1, 1]))
        self.default_min_size = self.params.get('min_size', 20)
        self.default_mask_threshold = self.params.get('mask_threshold', 0.53)
        self.default_peak_distance = int(self.params.get('peak_distance', 5))
        self.default_seed_threshold = self.params.get('seed_threshold', 0.1)
        self.default_overlap_threshold = self.params.get('overlap_threshold', 0.3)
        self.default_mean_threshold = self.params.get('mean_threshold', 0.0)
        self.default_fg_threshold = self.params.get('fg_threshold', 0.5)
        self.default_window_size = self.params.get('window_size',32) #32
        self.default_cleanup_fragments = self.params.get('cleanup_fragments', True)
        self.default_resolve_cell_and_nucleus = self.params.get('resolve_cell_and_nucleus', True)


    def forward(self, x: torch.Tensor,
                args: Optional[Dict[str, torch.Tensor]] = None,
                target_segmentation: torch.Tensor = torch.tensor([1, 1]), # Nuclei / Cells
                min_size: Optional[int] = None,
                mask_threshold: Optional[float] = None,
                peak_distance: Optional[int] = None,
                seed_threshold: Optional[float] = None,
                overlap_threshold: Optional[float] = None,
                mean_threshold: Optional[float] = None,
                fg_threshold: Optional[float] = None,
                window_size: Optional[int] = None,
                cleanup_fragments: Optional[bool] = None,
                resolve_cell_and_nucleus: Optional[bool] = None,
                precomputed_seeds: torch.Tensor = torch.tensor([]),
                ) -> torch.Tensor:
        
        min_size = int(min_size) if min_size is not None else self.default_min_size
        mask_threshold = float(mask_threshold) if mask_threshold is not None else self.default_mask_threshold
        peak_distance = int(peak_distance) if peak_distance is not None else self.default_peak_distance
        seed_threshold = float(seed_threshold) if seed_threshold is not None else self.default_seed_threshold
        overlap_threshold = float(overlap_threshold) if overlap_threshold is not None else self.default_overlap_threshold
        mean_threshold = float(mean_threshold) if mean_threshold is not None else self.default_mean_threshold
        fg_threshold = float(fg_threshold) if fg_threshold is not None else self.default_fg_threshold
        window_size = int(window_size) if window_size is not None else self.default_window_size
        cleanup_fragments = bool(cleanup_fragments) if cleanup_fragments is not None else self.default_cleanup_fragments
        resolve_cell_and_nucleus = bool(resolve_cell_and_nucleus) if resolve_cell_and_nucleus is not None else self.default_resolve_cell_and_nucleus

        if args is None:
            args = {"None": torch.tensor([0])}

        target_segmentation = args.get('target_segmentation', target_segmentation)
        min_size = int(args.get('min_size', torch.tensor(float(min_size))).item())
        mask_threshold = args.get('mask_threshold', torch.tensor(mask_threshold)).item()
        peak_distance = args.get('peak_distance', torch.tensor(peak_distance)).item()
        seed_threshold = args.get('seed_threshold', torch.tensor(seed_threshold)).item()
        overlap_threshold = args.get('overlap_threshold', torch.tensor(overlap_threshold)).item()
        mean_threshold = args.get('mean_threshold', torch.tensor(mean_threshold)).item()
        fg_threshold = args.get('fg_threshold', torch.tensor(fg_threshold)).item()
        window_size = int(args.get('window_size', torch.tensor(float(window_size))).item())
        cleanup_fragments = args.get('cleanup_fragments', torch.tensor(cleanup_fragments)).item()
        resolve_cell_and_nucleus = args.get('resolve_cell_and_nucleus', torch.tensor(resolve_cell_and_nucleus)).item()
        precomputed_seeds = args.get('precomputed_seeds', precomputed_seeds)

        torch.clamp_max_(x, 3) #Safety check, please normalize inputs properly!
        torch.clamp_min_(x, -2)


        x, pad = _instanseg_padding(x, extra_pad = 0)


        with torch.no_grad():


            x_full = self.fcn(x)

            dim_out = x_full.shape[1]

            if self.cells_and_nuclei:
                iterations = torch.tensor([0,1]) [target_segmentation.squeeze().to("cpu") > 0 ]
                dim_out = int(dim_out / 2)

            else:
                iterations = torch.tensor([0])
                dim_out = dim_out

            output_labels_list = []

            for image_index in range(x_full.shape[0]):
                labels_list = []
                for i in iterations:
                    if i == 0:
                        x = x_full[image_index,0: dim_out, :, :]
                    else:
                        x = x_full[image_index,dim_out:, :, :]

                    x = _recover_padding(x, pad)

                    height, width = x.size(1), x.size(2)

                    xxyy = generate_coordinate_map(mode = "linear", spatial_dim = self.dim_coords, height = height, width = width, device = x.device)


                    fields = (torch.sigmoid(x[0:self.dim_coords])-0.5) * 8


                    sigma = x[self.dim_coords:self.dim_coords + self.n_sigma]

                    #mask_map = torch.sigmoid(x[self.dim_coords + self.n_sigma]) #legacy
                    mask_map = ((x[self.dim_coords + self.n_sigma]) / 15) + 0.5 # inverse transform applied to edt during training.

                    if self.dim_seeds == 1:
                        mask_map = ((x[-1]) / 15) + 0.5
                        binary_map = mask_map
                    else:
                        binary_map = torch.sigmoid(x[-1])
                        mask_map = (x[-2] / 15).clone() + 0.5

                        mask_map[~(binary_map > fg_threshold)] = 0  # Set seed_map to 0 where binary_map is False


                    if precomputed_seeds is None or precomputed_seeds.shape[0] == 0:
                        centroids_idx = torch_peak_local_max(mask_map, neighbourhood_size=peak_distance,
                                                            minimum_value=seed_threshold, dtype= self.index_dtype)  # .to(prediction.device)
                    else:
                        centroids_idx = precomputed_seeds.to(mask_map.device).long()

                    fields = fields + xxyy

                    fields_at_centroids = fields[:, centroids_idx[:, 0], centroids_idx[:, 1]]

                    x = fields
                    c = fields_at_centroids.T
                    E = x.shape[0]
                    h, w = x.shape[-2:]
                    C = c.shape[0]
                    S = sigma.shape[0]

                    if C == 0:
                        label = torch.zeros(mask_map.shape, dtype= torch.float32, device=mask_map.device).squeeze()
                        labels_list.append(label)
                        continue

                    window_size = min(window_size, height, width)
                    centroids = centroids_idx.clone().cpu()  # C,2
                    centroids[:, 0].clamp_(min=window_size, max=h - window_size)
                    centroids[:, 1].clamp_(min=window_size, max=w - window_size)
                    window_slices = centroids[:, None].to(x.device) + torch.tensor([[-1, -1], [1, 1]] , device = x.device, dtype=centroids.dtype) * window_size
                    window_slices = window_slices  # C,2,2

                    slice_size = window_size * 2

                    # Create grids of indices for slice windows
                    grid_x, grid_y = torch.meshgrid(
                        torch.arange(slice_size, device=x.device, dtype=self.index_dtype),
                        torch.arange(slice_size, device=x.device, dtype=self.index_dtype), indexing="ij")
                    mesh = torch.stack((grid_x, grid_y))

                    mesh_grid = mesh.expand(C, 2, slice_size, slice_size)  # C,2,2*window_size,2*window_size
                    mesh_grid_flat = torch.flatten(mesh_grid, 2).permute(1, 0, -1)  # 2,C,2*window_size*2*window_size
                    idx = window_slices[:, 0].permute(1, 0)[:, :, None]
                    mesh_grid_flat = mesh_grid_flat + idx
                    mesh_grid_flat = torch.flatten(mesh_grid_flat, 1)  # 2,C*2*window_size*2*window_size

                #    x = self.traced_feature_engineering(x, c, sigma, torch.tensor(window_size).int(), mesh_grid_flat)
                    x = feature_engineering_slow(x, c, sigma, torch.tensor(window_size).int(), mesh_grid_flat)

                    x = torch.sigmoid(self.pixel_classifier(x))

                    x = x.reshape(C, 1, slice_size, slice_size)

                    coords = mesh_grid_flat.reshape(2, C, slice_size, slice_size)

                    if min_size > 0:
                        valid_objects = (x >= mask_threshold).sum((2,3)).squeeze() > min_size
                        x = x[valid_objects]
                        coords = coords[:,valid_objects]
                        mesh_grid_flat = coords.flatten(1)
                        centroids = centroids[valid_objects]
                        centroids_idx = centroids_idx[valid_objects]
                        window_slices = window_slices[valid_objects]

                    C = x.shape[0]

                    if C == 0:
                        label = torch.zeros(mask_map.shape, dtype= torch.float32, device=mask_map.device).squeeze()
                        labels_list.append(label)
                        continue

                    if cleanup_fragments:

                        top_left = window_slices[:,0,:]
                        shifted_centroid = centroids_idx - top_left

                        seeds = torch.zeros_like(x)
                        seeds[torch.arange(C, device=x.device), 0, shifted_centroid[:,0], shifted_centroid[:,1]] = 1

                        filled = flood_fill((x >= mask_threshold), seeds) > 0
                        holes = torch.bitwise_xor(fill_holes(filled), filled)
                    
                        x[~filled] = 0 #remove fragments
                        x[holes] = mask_threshold #fill holes


                    if x.is_mps:
                        device = 'cpu'
                        mesh_grid_flat = mesh_grid_flat.to(device)
                        x = x.to(device)
                        mask_map = mask_map.to(device)

                    labels = convert(x, coords, size=(h, w), mask_threshold=mask_threshold)[None]

                    idx = torch.arange(1, C + 1, device=x.device, dtype = self.index_dtype)
                    stack_ID = torch.ones((C, slice_size, slice_size), device=x.device, dtype=self.index_dtype)
                    stack_ID = stack_ID * (idx[:, None, None] - 1)

                    iidd = torch.stack((stack_ID.flatten(), mesh_grid_flat[0] * w + mesh_grid_flat[1]))

                    fg = x.flatten() >= mask_threshold
                    x = x.flatten()[fg]
                    sparse_onehot = torch.sparse_coo_tensor(
                        iidd[:, fg],
                        (x.flatten() >= mask_threshold).float(),
                        size=(C, h * w),
                        dtype=x.dtype,
                        device=x.device
                    )

                    iou = fast_sparse_iou(sparse_onehot)
                   # iou = fast_sparse_intersection_over_minimum_area(sparse_onehot)

                    remapping = find_connected_components((iou > overlap_threshold).to(self.index_dtype))
                    
                    labels = remap_values(remapping, labels)

                    labels_list.append(labels.squeeze())

                for i, lab in enumerate(labels_list):
                    if lab.is_mps:
                        labels_list[i] = lab.to("cpu")


                if len(labels_list) == 1:
                    lab = labels_list[0][None, None]  # 1,1,H,W
                else:
                    lab = torch.stack(labels_list)[None] 

                if lab.shape[1] == 2 and resolve_cell_and_nucleus: #nuclei and cells
                    lab = resolve_cell_and_nucleus_boundaries(lab)

                output_labels_list.append(lab[0])
            
            lab = torch.stack(output_labels_list) # B,C,H,W


            return lab.to(torch.float32) # B,C,H,W






