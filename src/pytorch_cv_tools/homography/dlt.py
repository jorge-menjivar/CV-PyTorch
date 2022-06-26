import torch


def dlTransformation(
    pairs: tuple[torch.Tensor, torch.Tensor],
    addr_left: list[list[int]],
    addr_right: list[list[int]],
):
    pairs_length = pairs[0].shape[0]
    A = torch.tensor(())
    for i in range(pairs_length):
        x, y = addr_right[int(pairs[0][i])]
        x_p, y_p = addr_left[int(pairs[1][i])]

        x_h = torch.mul(-x_p, torch.tensor([x, y, 1]))
        y_h = torch.mul(-y_p, torch.tensor([x, y, 1]))

        m_1 = torch.tensor([x, y, 1, 0, 0, 0, x_h[0], x_h[1], x_h[2]])
        m_2 = torch.tensor([0, 0, 0, x, y, 1, y_h[0], y_h[1], y_h[2]])
        A = torch.cat((A, m_1.unsqueeze(0)))
        A = torch.cat((A, m_2.unsqueeze(0)))

    svd2 = torch.linalg.svd(A)
    H = svd2.Vh[8]

    H_hat = H.reshape((3, 3))
    H_hat_normalized = H_hat / H_hat[2, 2]

    return H_hat_normalized
