import torch
import torch.nn.functional as F
from cv_pytorch.filters.gaussian import gaussianFilter


def bilateralFilter(image_tensor: torch.Tensor, sigmaS: float, sigmaI: float):
    r"""Bilaterally blurs the given image.

    Parameters
    ----------
    image : Image.Image
        Image to which the bilateral filter will be applied.
    
    sigmaS : (float)
        The size of the considered neighborhood.
    
    sigmaI : (float)
        The minimum amplitude of an edge.

    device: torch device to run the computations on.
    
    Returns
    ----------
    
    tensor : (torch.tensor)
        The blurred image as a tensor.
    """

    # C = image_tensor.shape[0]
    # H = image_tensor.shape[1]
    # W = image_tensor.shape[2]
    # R = g.shape[0]
    # S = g.shape[1]

    output = torch.zeros_like(image_tensor)

    # For each channel, apply filter
    for i, channel in enumerate(image_tensor[0]):
        output[i] = bilateralFilterSingleChannel(channel, sigmaS, sigmaI)

    output = output.clamp(0, 1)

    return output


def bilateralFilterSingleChannel(channel: torch.Tensor, sigmaS: float,
                                 sigmaI: float):

    H = channel.shape[0]
    W = channel.shape[1]

    g = gaussianFilter(sigmaS)

    R = g.shape[0]
    S = g.shape[1]

    pad_lr = S // 2
    pad_tb = R // 2

    # Add the necessary padding to apply the filter
    input = F.pad(channel, (pad_lr, pad_lr, pad_tb, pad_tb))

    output = torch.zeros((H, W))

    flat_gaussian = g.flatten()
    zeros = torch.zeros_like(flat_gaussian)

    for i in range(H):
        for j in range(W):

            x = j + S // 2
            y = i + R // 2
            main_pix = input[y, x]

            # Box coordinates
            t = i
            b = i + R
            l = j
            r = j + S

            # Getting the values in the box
            flat_box = input[t:b, l:r].flatten()

            lower_bound = main_pix - sigmaI
            upper_bound = main_pix + sigmaI

            # Applying threshold to the values in the box
            filter = torch.where(flat_box > lower_bound, flat_gaussian, zeros)

            filter = torch.where(flat_box < upper_bound, filter, zeros)

            # Normalizing
            filter = torch.divide(filter, filter.sum())

            # Applying normalized filter
            output[i, j] = torch.matmul(flat_box, filter)

    return output
