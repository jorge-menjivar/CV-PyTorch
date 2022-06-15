import torch
import torchvision.transforms as T
from PIL import Image


class NNScale:
    r"""Apply Nearest Neighbor Interpolation filter on image to scale it

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
        self.__width_scale = w_scale
        self.__height_scale = h_scale
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
                new_pixel = self.__nearestNeighbor(x, y)
                scaled_tensor[:, x, y] = new_pixel
        
        self.image = T.ToPILImage()(scaled_tensor)

        self.tensor = scaled_tensor

    def __nearestNeighbor(self, x: int, y: int):
        nnx = x // self.__width_scale
        nny = y // self.__height_scale

        return self.image_tensor[:, nnx, nny]
