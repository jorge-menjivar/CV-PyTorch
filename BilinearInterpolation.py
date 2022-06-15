import torch
import torchvision.transforms as T
from math import floor
from PIL import Image


def BilinearInterpolation(image_tensor: torch.Tensor, x: float, y: float):
    """Computes the value of a pixel if we scale the image.

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


class BilinearScale:
    r"""Apply Bilinear Interpolation filter on image to scale it

    Parameters
    ----------
    image : Image.Image
        Image to which the bilinear interpolation filter will be applied.
    
    w_scale : (float)
        The amount to scale the image in the x direction.
    
    h_scale : (float)
        The amount to scale the image in the y direction.

    device: torch device to run the computations on.
    
    Attributes
    ----------
    image : (PIL.Image.Image)
        The scaled image.
    
    tensor : (torch.tensor)
        The scaled image as a tensor.
    """

    def __init__(
        self,
        image: Image.Image,
        w_scale: float,
        h_scale: float,
        device: torch.device
    ):
        self.device = device
        image_tensor = T.ToTensor()(image).to(self.device)
        self.image_tensor = image_tensor
        C = image_tensor.shape[0]
        self.__old_W = image_tensor.shape[1]
        self.__old_H = image_tensor.shape[2]
        new_W = round(self.__old_W * w_scale)
        new_H = round(self.__old_H * h_scale)
        scaled_tensor = torch.zeros(
            (C, new_W, new_H),
            dtype=torch.float32,
            device=self.device
        )

        for x in range(new_W):
            for y in range(new_H):
                a = x % w_scale / w_scale
                b = y % h_scale / h_scale
                tmp_x = x // w_scale
                tmp_y = y // h_scale
                tmp_x = tmp_x + a
                tmp_y = tmp_y + b
                new_pixel = BilinearInterpolation(image_tensor, tmp_x, tmp_y)
                scaled_tensor[:, x, y] = new_pixel
        
        self.image = T.ToPILImage()(scaled_tensor)

        self.tensor = scaled_tensor
