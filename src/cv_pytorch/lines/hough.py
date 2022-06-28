from math import ceil, sqrt

import torch


def houghTransform(input: torch.Tensor):
    r"""Apply Hough Transform to find lines in the image
    Parameters
    ----------
    input : torch.Tensor
        `Canny` peaks tensor in the shape of (C, H, W), to which Hough
        Transform will be applied to.
    Returns
    ----------
    hough_space : (Torch.Tensor)
        The Hough space
    
    Notes
    -----
    - For each edge point x, y in the tensor
        - for theta = 0 to 180
            - r = xcos(theta) + ysin(theta)
            - H[theta, r] += 1
    - Find the values of (theta, r) where H[theta, r] is maximum
    - The detected line in the tensor is given by d = xcos(theta) + ysin(theta)
    - Using [scikit-image](https://github.com/scikit-image/scikit-image/blob/
    376d30f5ba83e98d2745622d9d4951914e0c91f7/skimage/transform/_hough_transform.pyx)
    as reference.
    """

    H = input.shape[1]
    W = input.shape[2]

    offset = ceil(sqrt(W**2 + H**2))
    hough_space_h = offset * 2 + 1
    hough_space_w = 180

    thetas = torch.linspace(-torch.pi / 2, torch.pi / 2, hough_space_w)
    thetas_size = thetas.shape[0]

    rhos = torch.linspace(-offset, offset, hough_space_h)

    hough_space = torch.zeros((hough_space_h, hough_space_w))

    n0_y, n0_x = torch.nonzero(input[0, :, :], as_tuple=True)
    n0_size = n0_x.shape[0]

    cosarr = torch.cos(thetas)
    sinarr = torch.sin(thetas)

    for i in range(n0_size):
        x = n0_x[i]
        y = n0_y[i]
        r = torch.round(torch.mul(x, cosarr) + torch.mul(y, sinarr))
        r = r + offset
        for j in range(thetas_size):
            hough_space[r[j].long(), j] += 1

    return hough_space, thetas, rhos


def findHoughLines(hough_space: torch.Tensor, threshold):
    r"""Find lines from the give Hough space
    Parameters
    ----------
    hough_space : torch.Tensor
        The hough space to find the lines from
    
    threshold: float
        The sensitivity to finding lines
    Returns
    ----------
    lines : list[Tensor]
        The found lines.
    """

    if threshold is None:
        threshold = 0.3

    threshold = threshold * torch.max(hough_space).item()

    lines: list[torch.Tensor] = []

    while (True):
        c_max = torch.argmax(hough_space)
        theta_index = c_max % hough_space.shape[1]
        rho = torch.div(c_max, hough_space.shape[1], rounding_mode='floor')
        val = hough_space[rho, theta_index]
        if val.item() < threshold:
            break
        else:
            lines.append(torch.tensor([[rho], [theta_index]]))
            hough_space[rho, theta_index] = 0

    return lines
