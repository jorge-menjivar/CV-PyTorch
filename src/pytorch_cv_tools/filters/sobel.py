import math
import torch
import torch.nn.functional as F
from scipy.stats import norm


def sobelFilter(image_tensor: torch.Tensor, sigma: float,
                device: torch.device):
    r"""Apply Sobel filter on image

    Parameters
    ----------
    image_tensor : torch.Tensor
        Image to which the sobel filter will be applied.

    device: torch device to run the computations on.

    Returns
    ----------
    magnitude : (torch.tensor)
        The magnitude image as a tensor.

    orientation : (torch.tensor)
        The orientation image as a tensor.

    Notes
    -----
    magnitude :math: g = \sqrt{gx^2 + gy^2}

    orientation :math: \Theta = tan^{-1}{gy/gx}
    """

    image_tensor = image_tensor.to(device).unsqueeze(0)

    f_wid = 4 * math.floor(sigma)
    G = norm.pdf(torch.arange(-f_wid, f_wid + 1), loc=0,
                 scale=sigma).reshape(-1, 1)
    G = torch.from_numpy(G)
    G = torch.mul(G.T, G).type(torch.float32)
    Gx, Gy = torch.gradient(G)

    gx = F.conv2d(image_tensor, Gx.unsqueeze(0).unsqueeze(0), padding='same')
    gy = F.conv2d(image_tensor, Gy.unsqueeze(0).unsqueeze(0), padding='same')

    # sqrt(gx^2 + gy^2)
    magnitude = torch.sqrt(torch.add(gx.pow(2), gy.pow(2)))

    # sqrt(pi^2 + pi^2)
    pi2 = torch.pi.__pow__(2)
    max_magnitude = torch.sqrt(torch.add(pi2, pi2))
    magnitude = torch.divide(magnitude, max_magnitude)

    orientation = torch.atan2(-gy, gx)

    # Scaling orientation output so it will be from 0 to 1
    orientation = orientation.add(torch.pi)
    orientation = orientation.div(2 * torch.pi)

    return magnitude, orientation
