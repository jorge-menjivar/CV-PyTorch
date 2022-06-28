import torch

from cv_pytorch.interpolation.nearest_neighbor import nearestNeighbor


def nnScale(
    input: torch.Tensor,
    w_scale: float,
    h_scale: float,
):
    """Apply Nearest Neighbor Interpolation filter on a tensor to scale it

    Parameters
    ----------
    input: torch.Tensor
        Tensor to which the nearest neighbot interpolation filter will be
        applied.
    
    w_scale : (float)
        The amount to scale the image in the x direction.
    
    h_scale : (float)
        The amount to scale the image in the y direction.
    
    Returns
    ----------
    output : (torch.tensor)
        The scaled tensor with nearest neighbor interpolation applied.
    """

    C = input.shape[0]
    old_W = input.shape[1]
    old_H = input.shape[2]
    new_W = round(old_W * w_scale)
    new_H = round(old_H * h_scale)

    output = torch.zeros((C, new_W, new_H), dtype=torch.float32)

    for x in range(new_W):
        for y in range(new_H):
            new_pixel = nearestNeighbor(input, x, y)
            output[:, x, y] = new_pixel

    return output
