from matplotlib import pyplot as plt
import torch


def getHoughLinesFigure(
    input: torch.Tensor,
    thetas: torch.Tensor,
    rhos: torch.Tensor,
    lines: list[torch.Tensor],
    cmap='viridis',
    figsize=None,
):
    r"""Return figure with the lines found using `houghTransform`
    and `findHoughLines` drawn on top

    Parameters
    ----------
    input : torch.Tensor
        The input image to draw the lines on

    thetas: torch.Tensor
        The angles returned from the `houghTransform` function

    rhos: torch.Tensor
        The lengths returned from the `houghTransform` function

    lines: list[torch.Tensor]
        The list of lines returned from the `findHoughLines` function

    cmap: str
        The color map to be used in the figure

    figsize: tuple[float, float] | None
        The size of the figure

    Returns
    ----------
    output : Figure
        A matplotlib figure that contains the input image with the given
        lines drawn on top.
    """

    plt.figure(figsize=figsize)
    plt.imshow(input.cpu(), cmap=cmap)
    plt.ylim(input.shape[1], 0)
    plt.xlim(0, input.shape[2])

    for line in lines:
        rho_index = line[0]
        rho = rhos[rho_index]
        theta_index = line[1]
        theta = thetas[theta_index]
        theta_normal = theta + (torch.pi / 2)
        x = torch.cos(theta) * rho
        y = torch.sin(theta) * rho
        slope = torch.tan(theta_normal).item()

        plt.axline((x.item(), y.item()), slope=slope)

    fig = plt.gcf()
    return fig


def printHoughLinesFigure(
    input: torch.Tensor,
    thetas: torch.Tensor,
    rhos: torch.Tensor,
    lines: list[torch.Tensor],
    cmap='viridis',
    figsize=None,
):
    r"""Print lines found using `houghTransform` and `findHoughLines`

    Parameters
    ----------
    input : torch.Tensor
        The input image to draw the lines on

    thetas: torch.Tensor
        The angles returned from the `houghTransform` function

    rhos: torch.Tensor
        The lengths returned from the `houghTransform` function

    lines: list[torch.Tensor]
        The list of lines returned from the `findHoughLines` function

    cmap: str
        The color map to be used in the figure

    figsize: tuple[float, float] | None
        The size of the figure

    Returns
    ----------
    None
    """

    plt.figure(figsize=figsize)
    plt.imshow(input.cpu(), cmap=cmap)
    plt.ylim(input.shape[1], 0)
    plt.xlim(0, input.shape[2])

    for line in lines:
        rho_index = line[0]
        rho = rhos[rho_index]
        theta_index = line[1]
        theta = thetas[theta_index]
        theta_normal = theta + (torch.pi / 2)
        x = torch.cos(theta) * rho
        y = torch.sin(theta) * rho
        slope = torch.tan(theta_normal).item()

        plt.axline((x.item(), y.item()), slope=slope)

    plt.show()

    return


def printTensor(input, cmap='viridis', figsize=None):
    r"""Print the given tensor

    Parameters
    ----------
    input : torch.Tensor
        The tensor to be printed

    cmap: str
        The color map to be used in the figure

    figsize: tuple[float, float] | None
        The size of the figure

    Returns
    ----------
    None
    """

    if input.ndim == 3:
        input = input.reshape(input.shape[1], input.shape[2], -1)

    plt.figure(figsize=figsize)
    plt.imshow(input.cpu().numpy(), cmap=cmap)
    plt.show()

    return
