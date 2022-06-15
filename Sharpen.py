import torch
import torchvision.transforms as T
from PIL import Image
from filters.gaussian import gaussianBlur


class Sharpen:
    """
    Sharpen an image.

    If a pixel is darker than the surrounding pixels, it will be
    darkened.

    If a pixel is lighter than the surrounding pixels, it will be
    lightened.

    Parameters
    ----------
    image : (PIL.Image.Image) An image file to run the filter on
    
    sigma : (float) The standard deviation of the Gaussian filter,
    which calculates the mean of the surrounding pixels.

    alpha : (float) The strength of the sharpening effect.

    device: torch device to run the computations on.

    separable: Whether to split the gaussian into separable
    x and y components to speed up computation.

    Attributes
    ----------
    image : (PIL.Image.Image)
        The processed image.
    
    tensor : (torch.tensor)
        The processed image as a tensor.

    Notes
    -----
    Filter is applied using
    :math: `original + alpha * (original - gaussian_smoothed)`
    """

    def __init__(
        self,
        image: Image.Image,
        sigma: float,
        alpha: float,
        device: torch.device
    ):

        self.device = device

        image_tensor = T.ToTensor()(image).to(self.device)
        sharpened = torch.zeros_like(image_tensor).to(self.device)
        
        smoothed = gaussianBlur(image_tensor, sigma, self.device)

        # oginal - gaussian_smoothed
        diff = torch.subtract(image_tensor, smoothed)
        
        # original + alpha * (original - gaussian_smoothed)
        sharpened = torch.add(image_tensor, torch.multiply(diff, alpha))

        # Clipping
        sharpened = torch.clamp(sharpened, 0, 1)

        sharpened_image: Image.Image = T.ToPILImage()(sharpened)

        self.image = sharpened_image
        self.tensor = sharpened

    def __getitem__(self, key):
        return getattr(self, key)
