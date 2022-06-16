import torch


def dist2(x: torch.Tensor, c: torch.Tensor):
    """
    Calculates squared distance between two sets of points.

    Parameters
    ----------
    x: torch.Tensor
        Data of shape `(ndata, dimx)`
    
    c: torch.Tensor
        Centers of shape `(ncenters, dimc)`

    Returns
    -------
    n2: torch.Tensor
        Squared distances between each pair of data from x and c, of shape
        `(ndata, ncenters)`
    
    Notes
    -----
    Adapted from Netlab neural network software:
    http://www.ncrg.aston.ac.uk/netlab/index.php

    This is a custom re-emplementation of matlab code originally written
    by Ian T Nabney

    Copyright (c) Ian T Nabney (1996-2001)
    """
    assert x.shape[1] == c.shape[1], \
        'Data dimension does not match dimension of centers'

    x = x.unsqueeze(0)  # new shape will be `(1, ndata, dimx)`
    c = c.unsqueeze(1)  # new shape will be `(ncenters, 1, dimc)`

    # We will now use broadcasting to easily calculate pairwise distances
    subs = torch.subtract(x, c)
    sqr = torch.pow(subs, 2)
    n2 = torch.sum(sqr, dim=-1)

    return n2
