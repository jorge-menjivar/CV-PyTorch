import torch
from math import floor


def bilinearInterpolation(image_tensor: torch.Tensor, x: float, y: float):
    """Computes the bilinearly interpolated value of a pixel.

    Parameters
    ----------
    new_x : int
        The x-position of the pixel to be computed
    new_y : int
        The y-position of the pixel to be computed

    Returns
    -------
    The value of pixel at position (new_x, new_y)

    Notes
    -----
    The value of the new pixel can be represented as
    :math: f(x+a,y+b) = (1-a)(1-b)f(x,y) + a(1-b)f(x+1,y) +
    (1-a)(b)f(x,y+1) + (ab)f(x+1,y+1)
    """

    W = image_tensor.shape[1]
    H = image_tensor.shape[2]

    a = x % 1
    b = y % 1
    x = floor(x)
    y = floor(y)

    # Setting cap
    x_plus_1 = min(x + 1, W - 1)
    y_plus_1 = min(y + 1, H - 1)

    v0 = torch.mul(image_tensor[:, x, y], (1 - a) * (1 - b))
    v1 = torch.mul(image_tensor[:, x_plus_1, y], a * (1 - b))
    v2 = torch.mul(image_tensor[:, x, y_plus_1], (1 - a) * b)
    v3 = torch.mul(image_tensor[:, x_plus_1, y_plus_1], a * b)

    val = torch.add(torch.add(torch.add(v0, v1), v2), v3)

    return val
