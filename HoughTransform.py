from math import ceil, sqrt

import torch
import torchvision.transforms as T
from PIL import Image
from matplotlib import pyplot as plt


class HoughTransform:
    r"""Apply Hough Transorm to find lines in the image

    Parameters
    ----------
    image : Image.Image
        Canny peaks image to which Hough Transform will be applied to.
    
    threshold: The lower bound for selecting lines. This number is
    multiplied by the maximum value in the hough space.

    device: torch device to run the computations on.

    Attributes
    ----------
    hough_space : (Torch.Tensor)
        The Hough space

    hough_space_image : (PIL.Image.Image)
        Image of the Hough space.
    
    hough_lines_image : (matplotlib.pyplot)
        A graph displaying the lines of the original image.
    
    Notes
    -----
    - For each edge point x,y in the image
        - for theta = 0 to 180
            - r = xcos(theta) + ysin(theta)
            - H[theta, r] += 1
    - Find the values of (theta, r) where H[theta, r] is maximum
    - The detected line in the image is given by d = xcos(theta) + ysin(theta)
    - Using [scikit-image](https://github.com/scikit-image/scikit-image/blob/
    376d30f5ba83e98d2745622d9d4951914e0c91f7/skimage/transform/_hough_transform.pyx)
    as reference.
    """

    def __init__(self,
                 image: Image.Image,
                 device: torch.device,
                 threshold=None):
        self.device = device

        image_tensor = T.ToTensor()(image).to(self.device)

        H = image_tensor.shape[1]
        W = image_tensor.shape[2]

        offset = ceil(sqrt(W**2 + H**2))
        hough_space_h = offset * 2 + 1
        hough_space_w = 180

        thetas = torch.linspace(-torch.pi / 2,
                                torch.pi / 2,
                                hough_space_w,
                                device=self.device)
        thetas_size = thetas.shape[0]

        rhos = torch.linspace(-offset,
                              offset,
                              hough_space_h,
                              device=self.device)

        hough_space = torch.zeros((hough_space_h, hough_space_w),
                                  device=self.device)

        n0_y, n0_x = torch.nonzero(image_tensor[0, :, :], as_tuple=True)
        n0_size = n0_x.shape[0]

        cosarr = torch.cos(thetas)
        sinarr = torch.sin(thetas)

        for i in range(n0_size):
            x = n0_x[i]
            y = n0_y[i]
            r = torch.round(torch.mul(x, cosarr) + torch.mul(y, sinarr))
            r = r + offset
            for j in range(thetas_size):
                hough_space[r[j].long(), j] += 1

        self.hough_space = hough_space
        self.hough_space_image: Image.Image = T.ToPILImage()(hough_space)

        self.hough_lines = self.__find_lines(hough_space, threshold)

        plt.imshow(image, cmap='gray')
        plt.ylim(image_tensor.shape[1], 0)
        plt.xlim(0, image_tensor.shape[2])

        for line in self.hough_lines:
            rho_index = line[0]
            rho = rhos[rho_index]
            theta_index = line[1]
            theta = thetas[theta_index]
            theta_normal = theta + (torch.pi / 2)
            x = torch.cos(theta) * rho
            y = torch.sin(theta) * rho
            slope = torch.tan(theta_normal).item()

            plt.axline((x.item(), y.item()), slope=slope)

        self.hough_lines_image = plt

    def __find_lines(self, hough_space: torch.Tensor, threshold):

        if threshold is None:
            threshold = 0.3

        threshold = threshold * torch.max(hough_space).item()

        lines = []

        while (True):
            c_max = torch.argmax(hough_space)
            theta_index = c_max % hough_space.shape[1]
            rho = torch.div(c_max, hough_space.shape[1], rounding_mode='floor')
            val = hough_space[rho, theta_index]
            if val.item() < threshold:
                break
            else:
                lines.append(torch.tensor([[rho], [theta_index]]))
                hough_space[rho, theta_index] = 0

        return lines

    def __getitem__(self, key):
        return getattr(self, key)
