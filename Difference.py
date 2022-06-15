import torch
import torchvision.transforms as T
from PIL import Image


def ImageDifference(
    image1: Image.Image,
    image2: Image.Image,
    device: torch.device
):
    '''
    Gets the difference in pixels between two images (image1 - image2).

    Parameters
    ----------
        image1 : (PIL.Image.Image)
            image1
        
        image2 : (PIL.Image.Image)
            image2
    
    Returns
    -------
        (torch.Tensor)
            Pixel differences between the images.
    '''

    image1_tensor: torch.Tensor = T.ToTensor()(image1).to(device)
    image2_tensor: torch.Tensor = T.ToTensor()(image2).to(device)

    diff = torch.subtract(image2_tensor, image1_tensor)

    absolute = torch.absolute(diff)

    print(f'Sum of Absolute: {torch.sum(absolute)}')

    return diff
