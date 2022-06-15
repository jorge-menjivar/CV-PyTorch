import torch
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image


class BilateralFilter:
    r"""Bilaterally blurs the given image.

    Parameters
    ----------
    image : Image.Image
        Image to which the bilinear interpolation filter will be applied.
    
    sigmaS : (float)
        The size of the considered neighborhood.
    
    sigmaI : (float)
        The minimum amplitude of an edge.

    device: torch device to run the computations on.
    
    Attributes
    ----------
    image : (PIL.Image.Image)
        The blurred image.
    
    tensor : (torch.tensor)
        The blurred image as a tensor.
    """

    def __init__(self, image: Image.Image, sigmaS: float, sigmaI: float,
                 device: torch.device):
        self.__device = device
        image_tensor = T.ToTensor()(image)
        self.__input = image_tensor

        self.__sigmaS = sigmaS
        self.__sigmaI = sigmaI
        self.__gaussian_filter = self.__createGaussianFilter()

        self.__C = self.__input.shape[0]
        self.__H = self.__input.shape[1]
        self.__W = self.__input.shape[2]
        self.__R = self.__gaussian_filter.shape[0]
        self.__S = self.__gaussian_filter.shape[1]

        output = torch.zeros_like(self.__input)

        for c in range(self.__C):
            output[c] = self.__apply_bf_to_channel(c)

        output = output.clamp(0, 1)
        self.tensor = output
        self.image: Image.Image = T.ToPILImage()(output)

    def __createGaussianFilter(self):
        sigma = self.__sigmaS
        filter_size = 2 * int(4 * sigma + 0.5) + 1
        gaussian_filter = torch.zeros(size=(filter_size, filter_size),
                                      dtype=torch.float32)

        for i in range(filter_size):
            for j in range(filter_size):
                x = i - filter_size // 2
                y = j - filter_size // 2
                Z = 2 * torch.pi * sigma**2
                e_x = torch.exp(torch.tensor(-(x**2 + y**2) / (2 * sigma**2)))
                gaussian_filter[i, j] = 1 / Z * e_x

        return gaussian_filter

    def __apply_bf_to_channel(self, channel: int):

        pad_lr = self.__S // 2
        pad_tb = self.__R // 2

        input = F.pad(self.__input, (pad_lr, pad_lr, pad_tb, pad_tb, 0, 0))
        output = torch.zeros((self.__H, self.__W))
        flat_gaussian = self.__gaussian_filter.flatten()
        zeros = torch.zeros_like(flat_gaussian)

        for i in range(self.__H):
            for j in range(self.__W):

                x = j + self.__S // 2
                y = i + self.__R // 2
                main_pix = input[channel, y, x]

                # Box coordinates
                t = i
                b = i + self.__R
                l = j
                r = j + self.__S

                # Getting the values in the box
                flat_box = input[channel, t:b, l:r].flatten()

                lower_bound = main_pix - self.__sigmaI
                upper_bound = main_pix + self.__sigmaI

                # Applying threshold to the values in the box
                filter = torch.where(flat_box > lower_bound, flat_gaussian,
                                     zeros)

                filter = torch.where(flat_box < upper_bound, filter, zeros)

                # Normalizing
                filter = torch.divide(filter, filter.sum())

                # Applying normalized filter
                output[i, j] = torch.matmul(flat_box, filter)

        return output
