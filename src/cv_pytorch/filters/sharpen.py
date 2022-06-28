import torch
from cv_pytorch.filters.gaussian import gaussianBlur


def sharpenFilter(input: torch.Tensor, sigma: float, alpha: float, device):
    """
    Apply sharpening filter to a tensor.

    If a pixel is darker than the surrounding pixels, it will be
    darkened.

    If a pixel is lighter than the surrounding pixels, it will be
    lightened.

    Parameters
    ----------
    input : torch.Tensor
        An tensor on which the filter will be applied
    
    sigma : (float)
        The standard deviation of the Gaussian filter,
        which calculates the mean of the surrounding pixels.

    alpha : (float)
        The strength of the sharpening effect.

    device: torch device to run the computations on.

    Returns
    ----------
    output : torch.Tensor
        The sharpened tensor

    Notes
    -----
    Filter is applied using
    :math: `original + alpha * (original - gaussian_smoothed)`
    """

    output = torch.zeros_like(input)

    smoothed = gaussianBlur(input, sigma, device)

    # oginal - gaussian_smoothed
    diff = torch.subtract(input, smoothed)

    # original + alpha * (original - gaussian_smoothed)
    output = torch.add(input, torch.multiply(diff, alpha))

    # Clipping
    output = torch.clamp(output, 0, 1)

    return output
