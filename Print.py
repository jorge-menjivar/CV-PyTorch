import torch
from PIL import Image as PILImage
import matplotlib.pyplot as plt
device = torch.device('cpu')


def Image(image: PILImage.Image, cmap='viridis', figsize=None):
    """
    Prints image using matplotlib
    """

    plt.figure(figsize=figsize)
    plt.imshow(image, cmap=cmap)
    plt.show()

    return


def Tensor(tensor, cmap='viridis', figsize=None):
    """
    Prints tensor using matplotlib
    """

    if tensor.ndim == 3:
        tensor = tensor.reshape(tensor.shape[1], tensor.shape[2], -1)
        
    plt.figure(figsize=figsize)
    plt.imshow(tensor.cpu().numpy(), cmap=cmap)
    plt.show()

    return
