import math
import torch
import torch.nn.functional as F
import numpy as np
from filters.sobel import sobel
from scipy.stats import norm
from scipy.ndimage import rank_filter


def harris(
    image_tensor: torch.Tensor,
    sigma: float,
    low_thresh=None,
    radius=None
):
    """
    Harris corner detector

    Parameters
    ----------
    image_tensor : torch.Tensor
        Grayscaled image tensor
    sigma : float
        Standard deviation of smoothing Gaussian
    thresh : float (optional)
    radius : float (optional)
        Radius of region considered in non-maximal suppression

    Returns
    -------
    corners : torch.Tensor
        Binary image marking corners
    y : torch.Tensor
        Row coordinates of corner points. Returned only if none of `thresh` and
        `radius` are None.
    x : torch.Tensor
        Column coordinates of corner points. Returned only if none of `thresh`
        and `radius` are None.

    Notes
    -----
    This is a custom re-emplementation of `Utils.harris()` using PyTorch.
    """

    image_tensor = image_tensor.type(torch.float32)

    # Adding a dimension in the beginning to comform with
    # (minibatch, channels, W, H) shape
    image_tensor = image_tensor.unsqueeze(0)

    # Repeats the given tensor pattern
    dx = torch.tile(
        torch.tensor([[-1, 0, 1]], dtype=torch.float32),
        dims=(3, 1)
    )

    # Transpose
    dy = dx.T

    Ix = F.conv2d(
        image_tensor,
        dx.unsqueeze(0).unsqueeze(0),
        padding='same'
    )
    Iy = F.conv2d(
        image_tensor,
        dy.unsqueeze(0).unsqueeze(0),
        padding='same'
    )

    f_wid = round(3 * math.floor(sigma))
    G = norm.pdf(
        torch.arange(-f_wid, f_wid + 1),
        loc=0,
        scale=sigma
    ).reshape(-1, 1)
    G = torch.from_numpy(G)
    G = torch.mul(G.T, G)
    G = torch.div(G, G.sum())
    G = G.unsqueeze(0).unsqueeze(0).type(torch.float32)

    Ix2 = F.conv2d(torch.pow(Ix, 2), G, padding='same').squeeze(0).squeeze(0)
    Iy2 = F.conv2d(torch.pow(Iy, 2), G, padding='same').squeeze(0).squeeze(0)
    Ixy = F.conv2d(torch.mul(Ix, Iy), G, padding='same').squeeze(0).squeeze(0)

    corners = torch.sub(torch.mul(Ix2, Iy2), torch.pow(Ixy, 2))
    corners = torch.div(corners, torch.add(Ix2, Iy2))

    if low_thresh is None or radius is None:
        return corners
    else:
        size = int(2 * radius + 1)
        mx = torch.tensor(rank_filter(corners, -1, size=size))
        corners: torch.Tensor = (corners == mx) & (corners > low_thresh)

        y, x = corners.nonzero(as_tuple=True)

        return corners, y, x


def dist2(x: torch.Tensor, c: torch.Tensor):
    """
    Calculates squared distance between two sets of points.

    Parameters
    ----------
    x: torch.Tensor
        Data of shape `(ndata, dimx)`
    c: torch.Tensor
        Centers of shape `(ncenters, dimc)`

    Returns
    -------
    n2: torch.Tensor
        Squared distances between each pair of data from x and c, of shape
        `(ndata, ncenters)`
    
    Notes
    -----
    This is a custom re-emplementation of `Utils.dist2()` using PyTorch.
    """
    assert x.shape[1] == c.shape[1], \
        'Data dimension does not match dimension of centers'

    x = x.unsqueeze(0)  # new shape will be `(1, ndata, dimx)`
    c = c.unsqueeze(1)  # new shape will be `(ncenters, 1, dimc)`

    # We will now use broadcasting to easily calculate pairwise distances
    subs = torch.subtract(x, c)
    sqr = torch.pow(subs, 2)
    n2 = torch.sum(sqr, dim=-1)

    return n2


def find_sift(
    image_tensor: torch.Tensor,
    circles: torch.Tensor,
    device,
    enlarge_factor=1.5
):
    """
    Compute non-rotation-invariant SIFT descriptors of a set of circles

    Parameters
    ----------
    image_tensor: torch.Tensor
        Image
    circles: torch.Tensor
        An array of shape `(ncircles, 3)` where ncircles is the number of
        circles, and each circle is defined by (x, y, r), where r is the radius
        of the cirlce
    enlarge_factor: float
        Factor which indicates by how much to enlarge the radius of the circle
        before computing the descriptor (a factor of 1.5 or large is usually
        necessary for best performance)

    Returns
    -------
    sift_arr: torch.Tensor
        Array of SIFT descriptors of shape `(ncircles, 128)`

    Notes
    -----
    This is a custom re-emplementation of `Utils.find_sift()` using PyTorch.
    """
    assert circles.ndim == 2 and circles.shape[1] == 3, \
        'Use circles array (keypoints array) of correct shape'
    image_tensor = image_tensor.type(torch.float32)

    if image_tensor.shape[0] == 1:
        image_tensor = image_tensor.squeeze(0)

    NUM_ANGLES = 8
    NUM_BINS = 4
    NUM_SAMPLES = NUM_BINS * NUM_BINS
    ALPHA = 9
    SIGMA_EDGE = 1

    ANGLE_STEP = 2 * np.pi / NUM_ANGLES
    angles = torch.arange(0, 2 * torch.pi, ANGLE_STEP)

    H = image_tensor.shape[0]
    W = image_tensor.shape[1]
    num_pts = circles.shape[0]

    sift_arr = torch.zeros((num_pts, NUM_SAMPLES * NUM_ANGLES))

    I_mag, I_theta = sobel(image_tensor, SIGMA_EDGE, device)

    # Sift bins
    interval = torch.arange(-1 + 1 / NUM_BINS, 1 + 1 / NUM_BINS, 2 / NUM_BINS)
    gridx, gridy = torch.meshgrid(interval, interval, indexing='ij')
    gridx = gridx.reshape((1, -1))
    gridy = gridy.reshape((1, -1))

    # Find Orientation
    I_orientation = torch.zeros((H, W, NUM_ANGLES))
    for i in range(NUM_ANGLES):
        tmp = torch.cos(I_theta - angles[i]) ** ALPHA
        tmp = tmp * (tmp > 0)

        I_orientation[:, :, i] = tmp * I_mag

    for i in range(num_pts):
        cx, cy = circles[i, :2]
        r = circles[i, 2] * enlarge_factor

        gridx_t = gridx * r + cx
        gridy_t = gridy * r + cy
        grid_res = 2.0 / NUM_BINS * r

        x_lo = int(
            torch.floor(torch.max(cx - r - grid_res / 2, torch.Tensor([0])))
        )
        x_hi = int(
            torch.ceil(torch.min(cx + r + grid_res / 2, torch.Tensor([W])))
        )
        y_lo = int(
            torch.floor(torch.max(cy - r - grid_res / 2, torch.Tensor([0])))
        )
        y_hi = int(
            torch.ceil(torch.min(cy + r + grid_res / 2, torch.Tensor([H])))
        )

        grid_px, grid_py = torch.meshgrid(
            torch.arange(x_lo, x_hi, 1),
            torch.arange(y_lo, y_hi, 1),
            indexing='ij'
        )
        grid_px = grid_px.reshape((-1, 1))
        grid_py = grid_py.reshape((-1, 1))

        dist_px = torch.sub(grid_px, gridx_t).abs()
        dist_py = torch.sub(grid_py, gridy_t).abs()

        weight_x = torch.div(
            dist_px, torch.add(grid_res, 1e-12)
        )
        weight_x = torch.mul(
            torch.sub(1, weight_x), (weight_x <= 1)
        )

        weight_y = torch.div(
            dist_py, torch.add(grid_res, 1e-12)
        )
        weight_y = torch.mul(
            torch.sub(1, weight_y), (weight_y <= 1)
        )

        weights = torch.mul(weight_x, weight_y)

        curr_sift = torch.zeros((NUM_ANGLES, NUM_SAMPLES))

        for j in range(NUM_ANGLES):
            tmp = I_orientation[y_lo:y_hi, x_lo:x_hi, j].reshape((-1, 1))
            curr_sift[j, :] = torch.mul(tmp, weights).sum(dim=0)
        sift_arr[i, :] = curr_sift.flatten()

    tmp = torch.sqrt(torch.sum(torch.pow(sift_arr, 2), dim=-1))
    if torch.sum(tmp > 1) > 0:
        sift_arr_norm = sift_arr[tmp > 1, :]
        sift_arr_norm /= tmp[tmp > 1].reshape(-1, 1)

        sift_arr_norm = torch.clamp(
            sift_arr_norm,
            min=sift_arr_norm.min(),
            max=torch.tensor([0.2])
        )

        sift_arr_norm /= torch.sqrt(
            torch.sum(torch.pow(sift_arr_norm, 2), dim=-1, keepdim=True)
        )

        sift_arr[tmp > 1, :] = sift_arr_norm

    return sift_arr
