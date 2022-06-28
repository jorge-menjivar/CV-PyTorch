import torch
import numpy as np

from cv_pytorch.filters.sobel import sobelFilter


def sifTransform(input: torch.Tensor,
                 circles: torch.Tensor,
                 device,
                 enlarge_factor=1.5):
    """
    Match features with SIFT

    Parameters
    ----------
    input: torch.Tensor
        The tensor that contains all matches
    
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
    output: torch.Tensor
        Array of SIFT descriptors of shape `(ncircles, 128)`

    Notes
    -----
    This is a custom re-emplementation of matlab code originally written
    by Lana Lazebnik

    Copyright (c) Lana Lazebnik
    """

    assert circles.ndim == 2 and circles.shape[1] == 3, \
        'Use circles array (keypoints array) of correct shape'
    input = input.type(torch.float32)

    if input.shape[0] == 1:
        input = input.squeeze(0)

    NUM_ANGLES = 8
    NUM_BINS = 4
    NUM_SAMPLES = NUM_BINS * NUM_BINS
    ALPHA = 9
    SIGMA_EDGE = 1

    ANGLE_STEP = 2 * np.pi / NUM_ANGLES
    angles = torch.arange(0, 2 * torch.pi, ANGLE_STEP)

    H = input.shape[0]
    W = input.shape[1]
    num_pts = circles.shape[0]

    output = torch.zeros((num_pts, NUM_SAMPLES * NUM_ANGLES))

    I_mag, I_theta = sobelFilter(input, SIGMA_EDGE, device)

    # Sift bins
    interval = torch.arange(-1 + 1 / NUM_BINS, 1 + 1 / NUM_BINS, 2 / NUM_BINS)
    gridx, gridy = torch.meshgrid(interval, interval, indexing='ij')
    gridx = gridx.reshape((1, -1))
    gridy = gridy.reshape((1, -1))

    # Find Orientation
    I_orientation = torch.zeros((H, W, NUM_ANGLES))
    for i in range(NUM_ANGLES):
        tmp = torch.cos(I_theta - angles[i])**ALPHA
        tmp = tmp * (tmp > 0)

        I_orientation[:, :, i] = tmp * I_mag

    for i in range(num_pts):
        cx, cy = circles[i, :2]
        r = circles[i, 2] * enlarge_factor

        gridx_t = gridx * r + cx
        gridy_t = gridy * r + cy
        grid_res = 2.0 / NUM_BINS * r

        x_lo = int(
            torch.floor(torch.max(cx - r - grid_res / 2, torch.Tensor([0]))))
        x_hi = int(
            torch.ceil(torch.min(cx + r + grid_res / 2, torch.Tensor([W]))))
        y_lo = int(
            torch.floor(torch.max(cy - r - grid_res / 2, torch.Tensor([0]))))
        y_hi = int(
            torch.ceil(torch.min(cy + r + grid_res / 2, torch.Tensor([H]))))

        grid_px, grid_py = torch.meshgrid(torch.arange(x_lo, x_hi, 1),
                                          torch.arange(y_lo, y_hi, 1),
                                          indexing='ij')
        grid_px = grid_px.reshape((-1, 1))
        grid_py = grid_py.reshape((-1, 1))

        dist_px = torch.sub(grid_px, gridx_t).abs()
        dist_py = torch.sub(grid_py, gridy_t).abs()

        weight_x = torch.div(dist_px, torch.add(grid_res, 1e-12))
        weight_x = torch.mul(torch.sub(1, weight_x), (weight_x <= 1))

        weight_y = torch.div(dist_py, torch.add(grid_res, 1e-12))
        weight_y = torch.mul(torch.sub(1, weight_y), (weight_y <= 1))

        weights = torch.mul(weight_x, weight_y)

        curr_sift = torch.zeros((NUM_ANGLES, NUM_SAMPLES))

        for j in range(NUM_ANGLES):
            tmp = I_orientation[y_lo:y_hi, x_lo:x_hi, j].reshape((-1, 1))
            curr_sift[j, :] = torch.mul(tmp, weights).sum(dim=0)
        output[i, :] = curr_sift.flatten()

    tmp = torch.sqrt(torch.sum(torch.pow(output, 2), dim=-1))
    if torch.sum(tmp > 1) > 0:
        sift_arr_norm = output[tmp > 1, :]
        sift_arr_norm /= tmp[tmp > 1].reshape(-1, 1)

        sift_arr_norm = torch.clamp(sift_arr_norm,
                                    min=sift_arr_norm.min(),
                                    max=torch.tensor([0.2]))

        sift_arr_norm /= torch.sqrt(
            torch.sum(torch.pow(sift_arr_norm, 2), dim=-1, keepdim=True))

        output[tmp > 1, :] = sift_arr_norm

    return output
