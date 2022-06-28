import torch

from cv_pytorch.interpolation.bilinear import bilinearInterpolation


def bilinearScale(input: torch.Tensor, w_scale: float, h_scale: float):
    r"""Scale tensor and apply bilinear interpolation to it.

    Parameters
    ----------
    input: torch.Tensor
        Tensor to which the bilinear interpolation filter will be applied.
    
    w_scale : (float)
        The amount to scale the image in the x direction.
    
    h_scale : (float)
        The amount to scale the image in the y direction.
    
    Returns
    ----------
    
    output : (torch.tensor)
        The scaled tensor with bilinear interpolation applied.
    """

    C = input.shape[0]
    old_W = input.shape[1]
    old_H = input.shape[2]
    new_W = round(old_W * w_scale)
    new_H = round(old_H * h_scale)

    output = torch.zeros((C, new_W, new_H), dtype=torch.float32)

    for x in range(new_W):
        for y in range(new_H):
            a = x % w_scale / w_scale
            b = y % h_scale / h_scale
            tmp_x = x // w_scale
            tmp_y = y // h_scale
            tmp_x = tmp_x + a
            tmp_y = tmp_y + b
            new_pixel = bilinearInterpolation(input, tmp_x, tmp_y)
            output[:, x, y] = new_pixel

    return output
