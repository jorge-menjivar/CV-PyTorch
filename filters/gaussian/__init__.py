import torch
import torch.nn.functional as F


def gaussianBlur(
    image_tensor: torch.Tensor,
    sigma: float,
    device: torch.device,
    separable: bool = True
):
    """Gaussian Blur an image.

    Parameters
    ----------
    image_tensor : torch.Tensor
        An image file to run the filter on.

    sigma : (float)
        The standard deviation of the Gaussian filter.
        The higher the sigma, the more blurred the image will become.

    device: torch device to run the computations on.

    separable: Whether to split the gaussian into separable
    x and y components to speed up computation.

    Returns
    ----------
    tensor : (torch.tensor)
        The blurred image as a tensor.
    """
    if separable:
        return __separableBlur(image_tensor, sigma, device)
    else:
        return __blur(image_tensor, sigma, device)

    
def gaussianFilter(sigma: float):
    filter_size = 2 * int(4 * sigma + 0.5) + 1
    gaussian_filter = torch.zeros(
        size=(filter_size, filter_size),
        dtype=torch.float32
    )

    for i in range(filter_size):
        for j in range(filter_size):
            x = i - filter_size // 2
            y = j - filter_size // 2
            Z = 2 * torch.pi * sigma**2
            e_x = torch.exp(torch.tensor(-(x**2 + y**2) / (2 * sigma**2)))
            gaussian_filter[i, j] = 1 / Z * e_x

    return gaussian_filter


def separableGaussianFilter(sigma: float):
    filter_size = 2 * int(4 * sigma + 0.5) + 1
    gaussian_filter_h = torch.zeros(
        size=(1, filter_size),
        dtype=torch.float32
    )
    gaussian_filter_v = torch.zeros(
        size=(filter_size, 1),
        dtype=torch.float32
    )

    for i in range(filter_size):
        for j in range(filter_size):
            x = i - filter_size // 2
            y = j - filter_size // 2
            Z = 2 * torch.pi * sigma**2
            e_x_h = torch.exp(torch.tensor(-(y**2) / (2 * sigma**2)))
            e_x_v = torch.exp(torch.tensor(-(x**2) / (2 * sigma**2)))
            gaussian_filter_h[0, j] = 1 / Z * e_x_h
            gaussian_filter_v[i, 0] = 1 / Z * e_x_v
    
    return gaussian_filter_h, gaussian_filter_v


def __blur(image_tensor: torch.Tensor, sigma: float, device):
    """ Blur an image with a 2D filter
    """

    image_tensor = image_tensor.to(device)
    gaussian_filter = gaussianFilter(sigma).to(device)

    gaussian = F.conv2d(
        image_tensor.unsqueeze(0),
        gaussian_filter.unsqueeze(0).unsqueeze(0),
        padding='same'
    )

    gaussian = gaussian.clamp(0, 1)

    return gaussian


def __separableBlur(image_tensor: torch.Tensor, sigma: float, device):
    """
    Gaussian Blur image through horizontal and vertical separation
    of Gaussian filter. Two 1D filters.
    """

    image_tensor = image_tensor.to(device)
    g_h, g_v = separableGaussianFilter(sigma)
    g_h = g_h.to(device)
    g_v = g_v.to(device)

    # Normalizing filter so the sum of h and v both add up to 1.
    g_h = g_h / g_h.sum()
    g_v = g_v / g_v.sum()

    f1 = F.conv2d(
        image_tensor.unsqueeze(0),
        g_h.unsqueeze(0).unsqueeze(0),
        padding='same'
    )

    gaussian = F.conv2d(
        f1.unsqueeze(0),
        g_v.unsqueeze(0).unsqueeze(0),
        padding='same'
    )

    gaussian = gaussian.clamp(0, 1)
    
    return gaussian
