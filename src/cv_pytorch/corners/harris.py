import torch
import torch.nn.functional as F
from scipy.stats import norm
from scipy.ndimage import rank_filter
import math


def harrisDetector(input: torch.Tensor,
                   sigma: float,
                   low_thresh=None,
                   radius=None):
    """
    Harris corner detector

    Parameters
    ----------
    input : torch.Tensor
        Grayscaled "image" tensor
    
    sigma : float
        Standard deviation of smoothing Gaussian filter.
    
    low_thresh : float (optional)
        The low bound sensitivity of which corners to return.

    radius : float (optional)
        Radius of region considered in non-maximal suppression

    Returns
    -------
    corners : torch.Tensor
        Binary tensor marking corners

    y : torch.Tensor
        Row coordinates of corner points. Returned only if none of `thresh` and
        `radius` are None.
    
    x : torch.Tensor
        Column coordinates of corner points. Returned only if none of `thresh`
        and `radius` are None.

    Notes
    -----
    Reference:
    C.G. Harris and M.J. Stephens. "A combined corner and edge detector",
    Proceedings Fourth Alvey Vision Conference, Manchester.
    pp 147-151, 1988.

    This is a custom re-emplementation of matlab code originally written
    by Peter Kovesi from Department of Computer Science & Software Engineering,
    The University of Western Australia
    pk@cs.uwa.edu.au  www.cs.uwa.edu.au/~pk
    """

    input = input.type(torch.float32)

    # Adding a dimension in the beginning to comform with
    # (minibatch, channels, W, H) shape
    input = input.unsqueeze(0)

    # Repeats the given tensor pattern
    dx = torch.tile(torch.tensor([[-1, 0, 1]], dtype=torch.float32),
                    dims=(3, 1))

    # Transpose
    dy = dx.T

    Ix = F.conv2d(input, dx.unsqueeze(0).unsqueeze(0), padding='same')
    Iy = F.conv2d(input, dy.unsqueeze(0).unsqueeze(0), padding='same')

    f_wid = round(3 * math.floor(sigma))
    G = norm.pdf(torch.arange(-f_wid, f_wid + 1), loc=0,
                 scale=sigma).reshape(-1, 1)
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
