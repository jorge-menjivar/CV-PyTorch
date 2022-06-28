from random import sample
import torch

from cv_pytorch.homography.dlt import dlTransformation


def ransac(pairs: tuple[torch.Tensor, torch.Tensor],
           addr_left: list[list[int]],
           addr_right: list[list[int]],
           inlier_upper_thres: float,
           epochs: int = 10):
    """
    Random Sample Consensus

    Returns
    -------

    Notes
    -----
    
    """

    pairs_length = pairs[0].shape[0]
    left_im = pairs[1]
    right_im = pairs[0]

    top_inlier_count = 0
    top_inliers = (torch.tensor([]), torch.tensor([]))
    avg_loss = 0
    # top_s_h = torch.tensor([])
    for i in range(epochs):
        # Sampling 4 pairs from all the pairs
        s_indices = sample(range(0, pairs_length), k=4)
        s_l = left_im[s_indices]
        s_r = right_im[s_indices]

        s_pairs = (s_r, s_l)
        s_h = dlTransformation(s_pairs, addr_left, addr_right)

        inliers, loss = __get_inliers(pairs, addr_left, addr_right, s_h,
                                      inlier_upper_thres)

        inlier_count = inliers[0].shape[0]

        if inlier_count > top_inlier_count:
            top_inliers = inliers
            top_inlier_count = inlier_count
            avg_loss = loss
            # top_s_h = s_h

    print(f'Inliers: {top_inlier_count}')
    print(f'Avg. Loss: {avg_loss}')

    H = dlTransformation(top_inliers, addr_left, addr_right)

    return H, top_inliers


def __get_inliers(
    pairs: tuple[torch.Tensor, torch.Tensor],
    addr_left: list[list[int]],
    addr_right: list[list[int]],
    homography: torch.Tensor,
    upper_thres: float,
):
    upper_thres = upper_thres**2
    pairs_length = pairs[0].shape[0]
    inliers_l = torch.tensor([])
    inliers_r = torch.tensor([])
    total_loss = 0
    for i in range(pairs_length):
        x, y = addr_right[pairs[0][i]]
        x_p, y_p = addr_left[pairs[1][i]]
        truth = torch.tensor([x_p, y_p])

        X = torch.tensor([x, y, 1], dtype=torch.float32)
        HX = torch.matmul(homography, X)

        x_pred = HX[0] / HX[2]
        y_pred = HX[1] / HX[2]

        pred = torch.tensor([x_pred, y_pred])

        loss = torch.subtract(pred, truth).abs().pow(2).sum()
        total_loss += loss

        if loss < upper_thres:
            inliers_l = torch.cat((inliers_l, pairs[1][i].unsqueeze(0)))
            inliers_r = torch.cat((inliers_r, pairs[0][i].unsqueeze(0)))

    # print(total_loss)
    avg_loss = total_loss / pairs_length
    return (inliers_r, inliers_l), avg_loss
