import torch
import torch.nn.functional as F
from typing import Tuple, Union
import numpy as np


def remap_values(remapping: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    This remaps the values in x according to the pairs in the remapping tensor.
    remapping: 2,N      Make sure the remapping is 1 to 1, and there are no loops (i.e. 1->2, 2->3, 3->1). Loops can be removed using graph based connected components algorithms (see instanseg postprocessing for an example)
    x: any shape
    """
    sorted_remapping = remapping[:, remapping[0].argsort()]
    index = torch.bucketize(x.ravel(), sorted_remapping[0])
    return sorted_remapping[1][index].reshape(x.shape)


# def torch_fastremap(x: torch.Tensor) -> torch.Tensor:
#    # if x.max() == 0:
#    #     return x
#   #  fg = x[x > 0]
#     unique_values = torch.unique(fg, sorted=True)
#     new_values = torch.arange(len(unique_values), dtype=x.dtype, device=x.device)
#     remapping = torch.stack((unique_values, new_values))
#     fg = remap_values(remapping, fg)
#     x[x > 0] = fg + 1
#     return x

def torch_fastremap(x: torch.Tensor) -> torch.Tensor:
    if x.max() == 0:
        return x
    unique_values = torch.unique(x, sorted=True)
    new_values = torch.arange(len(unique_values), dtype=x.dtype, device=x.device)
    remapping = torch.stack((unique_values, new_values))
    return remap_values(remapping, x)



def torch_onehot(x: torch.Tensor) -> torch.Tensor:
    # x is a labeled image of shape _,_,H,W returns a onehot encoding of shape 1,C,H,W

    if x.max() == 0:
        return torch.zeros_like(x).reshape(1, 0, *x.shape[-2:])
    H, W = x.shape[-2:]
    x = x.view(-1, 1, H, W)
    x = x.squeeze().view(1, 1, H, W)
    unique = torch.unique(x[x > 0])
    x = x.repeat(1, len(unique), 1, 1)
    return x == unique.unsqueeze(-1).unsqueeze(-1)


def fast_iou(onehot: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    # onehot is C,H,W
    if onehot.ndim == 3:
        onehot = onehot.flatten(1)
    onehot = (onehot > threshold).float()
    intersection = onehot @ onehot.T
    union = onehot.sum(1)[None].T + onehot.sum(1)[None] - intersection
    return intersection / union


def fast_sparse_iou(sparse_onehot: torch.Tensor) -> torch.Tensor:

    intersection = torch.sparse.mm(sparse_onehot, sparse_onehot.T).to_dense()
    sparse_sum = torch.sparse.sum(sparse_onehot, dim=(1,))[None].to_dense()
    union = sparse_sum.T + sparse_sum - intersection
    return intersection / union

def fast_sparse_intersection_over_minimum_area(sparse_onehot: torch.Tensor) -> torch.Tensor:
    """
    Computes the sparse Intersection over Minimum Area (IoMA) for a given boolean sparse one-hot tensor.

    Args:
        sparse_onehot (torch.Tensor): A boolean sparse tensor of shape (N, M).

    Returns:
        torch.Tensor: A dense tensor of shape (N, N) containing the IoMA values.
    """
    # Compute intersection
    intersection = torch.sparse.mm(sparse_onehot, sparse_onehot.T).to_dense()
    
    # Compute the area (sum of ones for each row)
    sparse_sum = torch.sparse.sum(sparse_onehot, dim=(1,)).to_dense()
    
    # Compute the minimum area for each pair
    min_area = torch.min(sparse_sum[:, None], sparse_sum[None, :])
    
    # Compute Intersection over Minimum Area
    return intersection / min_area


def instance_wise_edt(x: torch.Tensor, edt_type: str = 'auto') -> torch.Tensor:
    """
    Create instance-normalized distance map from a labeled image.
    Each pixel within an instance gives the distance to the closed background pixel,
    divided by the maximum distance (so that the maximum within an instance is 1).

    The calculation of the Euclidean Distance Transform can use the 'edt' or 'monai'
    packages.
    'edt' is faster for CPU computation, while 'monai' can use cucim for GPU acceleration
    where CUDA is available.
    Use 'auto' to decide automatically.
    """
    if x.max() == 0:
        return torch.zeros_like(x).squeeze()
    is_mps = x.is_mps
    if is_mps:
        # Need to convert to CPU for MPS, because distance transform gives float64 result
        # and Monai's internal attempt to convert type will fail
        x = x.to('cpu')

    use_edt = edt_type == 'edt' or (edt_type != 'monai' and not x.is_cuda)
    if use_edt:
        import edt
        xedt = torch.from_numpy(edt.edt(x[0].cpu().numpy(), black_border=False))
        x = torch_onehot(x)[0] * xedt.to(x.device)
    else:
        import monai
        x = torch_onehot(x)
        x = monai.transforms.utils.distance_transform_edt(x[0])

    # Normalize instance distances to have max 1
    x = x / (x.flatten(1).max(1)[0]).view(-1, 1, 1)
    x = x.sum(0)

    if is_mps:
        x = x.type(torch.FloatTensor).to('mps')
    return x



def fast_dual_iou(onehot1: torch.Tensor, onehot2: torch.Tensor) -> torch.Tensor:
    """
    Returns the intersection over union between two dense onehot encoded tensors
    """
    # onehot1 and onehot2 are C1,H,W and C2,H,W

    C1 = onehot1.shape[0]
    C2 = onehot2.shape[0]

    max_C = max(C1, C2)

    onehot1 = torch.cat((onehot1, torch.zeros((max_C - C1, *onehot1.shape[1:]))), dim=0)
    onehot2 = torch.cat((onehot2, torch.zeros((max_C - C2, *onehot2.shape[1:]))), dim=0)

    onehot1 = onehot1.flatten(1)
    onehot1 = (onehot1 > 0.5).float()  # onehot should be binary

    onehot2 = onehot2.flatten(1)
    onehot2 = (onehot2 > 0.5).float()

    intersection = onehot1 @ onehot2.T
    union = (onehot1).sum(1)[None].T + (onehot2).sum(1)[None] - intersection

    return (intersection / union)[:C1, :C2]


def torch_sparse_onehot(x: torch.Tensor, flatten: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    # x is a labeled image of shape _,_,H,W returns a sparse tensor of shape C,H,W
    unique_values = torch.unique(x, sorted=True)
    x = torch_fastremap(x)

    H, W = x.shape[-2], x.shape[-1]


    if flatten:

        if x.max() == 0:
            return torch.zeros_like(x).reshape(1, 1, H*W)[:,:0] , unique_values
        

        x = x.reshape(H * W)
        xxyy = torch.nonzero(x > 0).squeeze(1)
        zz = x[xxyy] - 1
        C = x.max().int().item()

       # print(C, H, W, type(C), type(H), type(W))
        sparse_onehot = torch.sparse_coo_tensor(torch.stack((zz, xxyy)).long(), (torch.ones_like(xxyy).float()),
                                                size=(int(C), int(H * W)), dtype=torch.float32)

    else:
        if x.max() == 0:
            return torch.zeros_like(x).reshape(1, 0, H,W) , unique_values

        x = x.squeeze().view(H, W)
        x_temp= torch.nonzero(x > 0).T
        zz = x[x_temp[0], x_temp[1]] - 1
        C = x.max().int().item()
        sparse_onehot = torch.sparse_coo_tensor(torch.stack((zz, x_temp[0], x_temp[1])).long(), (torch.ones_like(x_temp[0]).float()),
                                                size=(int(C), int(H), int(W)), dtype=torch.float32)

    return sparse_onehot, unique_values


import pdb


def fast_sparse_dual_iou(onehot1: torch.Tensor, onehot2: torch.Tensor) -> torch.Tensor:
    """
    Returns the (dense) intersection over union between two sparse onehot encoded tensors
    """
    # onehot1 and onehot2 are C1,H*W and C2,H*W

    intersection = torch.sparse.mm(onehot1, onehot2.T).to_dense()
    sparse_sum1 = torch.sparse.sum(onehot1, dim=(1,))[None].to_dense()
    sparse_sum2 = torch.sparse.sum(onehot2, dim=(1,))[None].to_dense()
    union = sparse_sum1.T + sparse_sum2 - intersection

    return (intersection / union)


def iou_test():
    """
    Unit test for the fast dual iou functions
    """
    out = torch.randint(0, 50, (1, 2, 124, 256), dtype=torch.float32)
    onehot1 = torch_onehot(out[0, 0])[0]
    onehot2 = torch_onehot(out[0, 1])[0]
    iou_dense = fast_dual_iou(onehot1, onehot2)

    onehot1 = torch_sparse_onehot(out[0, 0], flatten=True)[0]
    onehot2 = torch_sparse_onehot(out[0, 1], flatten=True)[0]
    iou_sparse = fast_sparse_dual_iou(onehot1, onehot2)

    assert torch.allclose(iou_dense, iou_sparse)



def match_labels(tile_1: torch.Tensor,tile_2: torch.Tensor,threshold: float = 0.5, strict = False):

    """This function takes two labeled tiles, and matches the overlapping labels of tile_2 to the labels of tile_1.
        If strict is set to True, the function will discard non matching objects.
       """
    
    if tile_1.max() == 0 or tile_2.max() == 0:
        if not strict:
            return tile_1, tile_2
        else:
            return torch.zeros_like(tile_1), torch.zeros_like(tile_2)
        
    old_problematic_onehot, old_unique_values = torch_sparse_onehot(tile_1, flatten=True)
    new_problematic_onehot, new_unique_values = torch_sparse_onehot(tile_2, flatten=True)

    iou = fast_sparse_dual_iou(old_problematic_onehot, new_problematic_onehot)

    onehot_remapping = torch.nonzero(iou > threshold).T# + 1

    if old_unique_values.min() == 0:
       old_unique_values = old_unique_values[old_unique_values > 0]
    if new_unique_values.min() == 0:
       new_unique_values = new_unique_values[new_unique_values > 0]


    if onehot_remapping.shape[1] > 0:
        
        onehot_remapping = torch.stack((new_unique_values[onehot_remapping[1]], old_unique_values[onehot_remapping[0]]))

        if not strict:
            mask = torch.isin(tile_2, onehot_remapping[0])
            tile_2[mask] = remap_values(onehot_remapping, tile_2[mask])

            return tile_1, tile_2
        else:
            tile_1 = tile_1 * torch.isin(tile_1, onehot_remapping[1]).int()
            tile_2 = tile_2 * torch.isin(tile_2, onehot_remapping[0]).int()

            tile_2[tile_2>0] = remap_values(onehot_remapping, tile_2[tile_2>0])

            return tile_1, tile_2
        
    else:
        if not strict:
            return tile_1, tile_2
        else:
            return torch.zeros_like(tile_1), torch.zeros_like(tile_2)
        

def connected_components(x: torch.Tensor, num_iterations: int = 32) -> torch.Tensor:
    """
    This function takes a binary image and returns the connected components
    """
    mask = x == 1

    B, _, H, W = x.shape
    out = torch.arange(B * W * H, device=x.device, dtype=x.dtype).reshape((B, 1, H, W))
    out[~mask] = 0

    for _ in range(num_iterations):
        out[mask] = F.max_pool2d(out, kernel_size=3, stride=1, padding=1)[mask]

    return out


def iou_heatmap(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    x is 1,1,H,W
    y is 1,1,H,W
    This function takes two labeled images and returns the intersection over union heatmap
    """
    if x.max() ==0 or y.max() == 0:
        return torch.zeros_like(x)

    x = torch_fastremap(x)
    y = torch_fastremap(y)

    x_onehot, _ = torch_sparse_onehot(x, flatten=True)
    y_onehot, _ = torch_sparse_onehot(y, flatten=True)

    iou = fast_sparse_dual_iou(x_onehot, y_onehot)
    predicted_iou = iou.sum(1)
    onehot = torch_onehot(x)
    onehot = onehot.float() * predicted_iou[:,None, None]
    map = onehot.max(1)[0]

    return map


def centroids_from_lab(lab: torch.Tensor):
    mesh_grid = torch.stack(torch.meshgrid(torch.arange(lab.shape[-2], device = lab.device), torch.arange(lab.shape[-1],device = lab.device), indexing="ij")).float()

    sparse_onehot, label_ids = torch_sparse_onehot(lab, flatten=True)

    sum_centroids = torch.sparse.mm(sparse_onehot, mesh_grid.flatten(1).T)

    centroids = sum_centroids / torch.sparse.sum(sparse_onehot, dim=(1,)).to_dense().unsqueeze(-1)

    return centroids, label_ids  # N,2  N


def get_patches(lab: torch.Tensor, image: torch.Tensor, patch_size: int = 64, return_lab_ids: bool = False):
    # lab is 1,H,W with N objects
    # image is C,H,W

    # Returns N,C,patch_size,patch_size

    centroids, label_ids = centroids_from_lab(lab)
    N = centroids.shape[0]

    C, h, w = image.shape[-3:]

    window_size = patch_size // 2
    centroids = centroids.clone()  # N,2
    centroids[:, 0] = centroids[:,0].clamp(min=window_size, max=h - window_size)
    centroids[:, 1] = centroids[:,1].clamp(min=window_size, max=w - window_size)
    window_slices = centroids[:, None] + torch.tensor([[-1, -1], [1, 1]]).to(image.device) * window_size
    window_slices = window_slices.long()  # N,2,2

    slice_size = window_size * 2

    # Create grids of indices for slice windows
    grid_x, grid_y = torch.meshgrid(
        torch.arange(slice_size, device=image.device),
        torch.arange(slice_size, device=image.device), indexing="ij")
    mesh = torch.stack((grid_x, grid_y))

    mesh_grid = mesh.expand(N, 2, slice_size, slice_size)  # N,2,2*window_size,2*window_size
    mesh_flat = torch.flatten(mesh_grid, 2).permute(1, 0, -1)  # 2,N,2*window_size*2*window_size
    idx = window_slices[:, 0].permute(1, 0)[:, :, None]
    mesh_flat = mesh_flat + idx
    mesh_flater = torch.flatten(mesh_flat, 1)  # 2,N*2*window_size*2*window_size


    out = image[:, mesh_flater[0], mesh_flater[1]].reshape(C, N, -1)
    out = out.reshape(C, N, patch_size, patch_size)
    out = out.permute(1, 0, 2, 3)


    if return_lab_ids:
        return out, label_ids

    return out,label_ids  # N,C,patch_size,patch_size


def get_masked_patches(lab: torch.Tensor, image: torch.Tensor, patch_size: int = 64):
    # lab is 1,H,W
    # image is C,H,W

    # if lab.max() == 0:
    #     if return_mask:
    #         return None,None
    #     return None

    lab_patches, label_ids = get_patches(lab, lab[0], patch_size)
    mask_patches = lab_patches == label_ids[1:, None, None, None]

    image_patches,_ = get_patches(lab, image, patch_size)

    # canvas = torch.ones_like(image_patches) * (~mask_patches).float()

    # image_patches = image_patches * mask_patches.float() + canvas

   # pdb.set_trace()

    return image_patches,mask_patches  # N,C,patch_size,patch_size



def eccentricity_batch(mask_tensor):
    """
    Calculate the eccentricity of a batch of binary masks. B,H,W -> returns B
    """
    
    # Get dimensions
    batch_size, m, n = mask_tensor.shape
    
    # Create indices grid
    y_indices, x_indices = torch.meshgrid(torch.arange(m), torch.arange(n), indexing='ij')
    y_indices = y_indices.unsqueeze(0).to(mask_tensor.device).expand(batch_size, m, n)
    x_indices = x_indices.unsqueeze(0).to(mask_tensor.device).expand(batch_size, m, n)
    
    # Find total mass and centroid
    total_mass = mask_tensor.sum(dim=(1, 2))
    centroid_y = (y_indices * mask_tensor).sum(dim=(1, 2)) / total_mass
    centroid_x = (x_indices * mask_tensor).sum(dim=(1, 2)) / total_mass
    
    # Calculate second-order moments
    y_diff = y_indices - centroid_y.view(batch_size, 1, 1)
    x_diff = x_indices - centroid_x.view(batch_size, 1, 1)
    M_yy = torch.sum(y_diff**2 * mask_tensor, dim=(1, 2))
    M_xx = torch.sum(x_diff**2 * mask_tensor, dim=(1, 2))
    M_xy = torch.sum(x_diff * y_diff * mask_tensor, dim=(1, 2))

    # Construct second-order moments tensor
    moments_tensor = torch.stack([torch.stack([M_xx, M_xy]),
                                  torch.stack([M_xy, M_yy])]).permute(2,0,1)
    

    # Compute eigenvalues
    eigenvalues = torch.linalg.eigvals(moments_tensor)

    # Get maximum eigenvalue
    lambda1 = torch.max(eigenvalues.real, dim=1).values
    # Get minimum eigenvalue
    lambda2 = torch.min(eigenvalues.real, dim=1).values
    
    # Calculate eccentricity
    eccentricity = torch.sqrt(1 - (lambda2 / lambda1))
    
    return eccentricity.squeeze(1,2)



def _to_ndim(x: torch.Tensor, n: int) -> torch.Tensor:
    """
    Ensure that the input tensor has the desired number of dimensions.
    If the input tensor has fewer dimensions, it will be unsqueezed.
    If the input tensor has more dimensions, it will be squeezed.
    If the input tensor has the desired number of dimensions, it will be returned as is.
    
    Args:
        x (torch.Tensor): The input tensor.
        n (int): The desired number of dimensions.
        
    Returns:
        torch.Tensor: The input tensor with the desired number of dimensions.
    """

    if x.dim() == n:
        return x
    if x.dim() > n:
        x = x.squeeze()
    x = x[(None,) * (n - x.dim())]
    if x.dim() != n:
        raise ValueError(f"Input tensor has shape {x.shape}, which is not compatible with the desired dimension {n}.")
    return x

def _to_ndim_numpy(x: np.ndarray, n: int) -> np.ndarray:
    """
    Ensure that the input NumPy array has the desired number of dimensions.
    If the input array has fewer dimensions, it will be unsqueezed.
    If the input array has more dimensions, it will be squeezed.
    If the input array has the desired number of dimensions, it will be returned as is.
    
    Args:
        x (np.ndarray): The input NumPy array.
        n (int): The desired number of dimensions.
        
    Returns:
        np.ndarray: The input NumPy array with the desired number of dimensions.
    """
    if x.ndim == n:
        return x
    if x.ndim > n:
        x = x.squeeze()
    x = x[(None,) * (n - x.ndim)]
    if x.ndim != n:
        raise ValueError(f"Input tensor has shape {x.shape}, which is not compatible with the desired dimension {n}.")
    return x


def _to_tensor_float32(image: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """
    Convert the input image to a PyTorch tensor with float32 data type.
    If the input is a NumPy array, it will be converted to a PyTorch tensor.
    The tensor will be squeezed to remove any singleton dimensions.
    The channel dimension will be moved to the first position if it is not already there.
    
    Args:
        image (Union[np.ndarray, torch.Tensor]): The input image, which can be either a NumPy array or a PyTorch tensor.
        
    Returns:
        torch.Tensor: The input image as a PyTorch tensor with float32 data type and the channel dimension in the first position.
    """

    if isinstance(image, np.ndarray):      
        if image.dtype == np.uint16:
            image = image.astype(np.int32)
        image = torch.from_numpy(image.astype(np.float32)).float()
    
    image = image.squeeze()

    assert image.dim() <= 3 and image.dim() >= 2, f"Input image shape {image.shape()} is not supported."

    image = torch.atleast_3d(image)
    channel_index = np.argmin(image.shape) #Note, this could break for small, highly multiplexed images.
    if channel_index != 0:
        image = image.movedim(channel_index, 0)

    return image


from instanseg.utils.utils import show_images
def flood_fill(bw_mask: torch.Tensor, bw_seed: torch.Tensor):

    bw_mask = _to_ndim(bw_mask, 4).clone().float()
    bw_seed = _to_ndim(bw_seed, 4).clone()
    
    max_iterations = max(bw_seed.shape[-1], bw_seed.shape[-2])
    for ii in range(max_iterations):
        bw_seed2 = torch.nn.functional.max_pool2d(bw_seed.float(), kernel_size=3, stride=1, padding=1)
        bw_seed2 = torch.bitwise_and(bw_seed2 > 0, bw_mask > 0)
        if torch.equal(bw_seed, bw_seed2):
            return bw_seed2 > 0
        bw_seed = bw_seed2
    print('Reached maximum number of iterations - this is not expected!')

    return bw_seed > 0

def fill_holes(bw_mask: torch.Tensor):
    bw_mask = _to_ndim(bw_mask, 4)
    bw_seed = dilate(bw_mask, mask = torch.ones_like(bw_mask), num_iterations = 3) > 0
    return ~flood_fill(~bw_mask, ~bw_seed)


def dilate(x: torch.Tensor, mask, num_iterations: int = 3) -> torch.Tensor:
    original_dim = x.dim()
    x = _to_ndim(x, 4).float()
    mask = _to_ndim(mask, 4)

    for _ in range(num_iterations):
        x[mask] = torch.nn.functional.max_pool2d(x.float(), kernel_size=3, stride=1, padding=1)[mask]
    return _to_ndim(x, original_dim)

def find_boundaries_max_pool_labeled(labeled_image: torch.Tensor) -> torch.Tensor:
    labeled_image = _to_ndim(labeled_image, 4).float()
    max_pooled = F.max_pool2d(labeled_image.float(), kernel_size=3, stride=1, padding=1)
    # Boundaries are where the max-pooled result differs from the original labels
    boundaries = (max_pooled != labeled_image.float()).float()
    return boundaries > 0

def find_hard_boundaries(labeled_image: torch.Tensor) -> torch.Tensor:
    all_boundaries = find_boundaries_max_pool_labeled(labeled_image)
    return((all_boundaries)* (labeled_image > 0)) > 0 


def expand_labels_map(labeled_image: torch.Tensor, num_iterations: int = 5) -> torch.Tensor:

    original_dim = labeled_image.dim()
    labeled_image = _to_ndim(labeled_image, 4)
    for _ in range(num_iterations):
        valid_region = (~ find_hard_boundaries(labeled_image))
        labeled_image[valid_region] = F.max_pool2d(labeled_image.float(), kernel_size=3, stride=1, padding=1)[valid_region]
    labeled_image = _to_ndim(labeled_image, original_dim)

    return labeled_image
