import torch


def nearestNeighbor(input: torch.Tensor, x: float, y: float):
    """Computes the nearest neighbor interpolation value of an
    element at position (x, y), where x and y are both continous
    values.

    Parameters
    ----------
    input: torch.Tensor
        Tensor that contains all the values.
    
    x : float
        The x-position of the location to be computed
        
    y : float
        The y-position of the location to be computed

    Returns
    -------
    The value of element at position (x, y)
    """

    H = input.shape[1]
    W = input.shape[2]

    nnx = x // W
    nny = y // H

    return input[:, nnx, nny]
