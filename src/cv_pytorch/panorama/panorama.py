from matplotlib import pyplot as plt
import numpy as np
from cv_pytorch.corners.harris import harrisDetector
from cv_pytorch.feature_matching.sift import sifTransform
from cv_pytorch.utils.difference import dist2
from cv_pytorch.utils.runtime import printRuntime
import torch
import torchvision.transforms as T
from PIL import Image, ImageOps
from cv_pytorch.outliers.ransac import ransac

from skimage.transform import warp

device = torch.device('cpu')


def __perform_harris(image, sigma, thresh, radius):
    grayscale = ImageOps.grayscale(image)
    image_tensor = T.ToTensor()(grayscale)

    corners, y, x = harrisDetector(image_tensor, sigma, thresh, radius)
    return corners, y, x


def __find_neighborhoods(input, radius, thresh):
    H = input.shape[0]
    W = input.shape[1]

    n0_y, n0_x = torch.nonzero(input, as_tuple=True)
    n0_size = n0_x.shape[0]

    neighborhoods = torch.Tensor([])
    addresses: list[list[int]] = []

    for i in range(n0_size):
        x = int(n0_x[i].item())
        y = int(n0_y[i].item())

        l = x - radius
        r = x + radius
        t = y - radius
        b = y + radius

        if l >= 0 and t >= 0 and r <= W - 1 and b <= H - 1:
            tmp = input[t:b, l:r].flatten().unsqueeze(0)
            neighborhoods = torch.cat((neighborhoods, tmp))

            addresses.append([x, y])

    return neighborhoods, addresses


def __find_matches(distance: torch.Tensor, upper_thresh: float):

    zeros = torch.zeros_like(distance)
    ones = torch.ones_like(distance)

    matches = torch.where(distance < upper_thresh, ones, zeros)

    return torch.nonzero(matches, as_tuple=True)


def __merge_images(image_1: torch.Tensor, image_2: torch.Tensor,
                   homography: torch.Tensor):

    if image_1.ndim == 3:
        image_1 = image_1.squeeze(0)

    if image_2.ndim == 3:
        image_2 = image_2.squeeze(0)

    # Converting zeros to nan
    image_1[image_1 == 0] = float('nan')
    image_2[image_2 == 0] = float('nan')

    channels = []
    for i in range(image_1.shape[0]):
        im1 = image_1[i, :, :]
        im2 = image_2[i, :, :]
        trfm_list = [np.eye(3), homography.inverse().cpu().numpy()]

        img_list = [im1.cpu().numpy(), im2.cpu().numpy()]

        margin_h = 1000
        margin_v = 1000
        height, width = img_list[0].shape
        out_shape = height + 2 * margin_v, width + 2 * margin_h
        glob_trfm = np.eye(3)
        glob_trfm[:2, 2] = -margin_h, -margin_v

        global_img_list = [
            warp(img,
                 -trfm.dot(glob_trfm),
                 output_shape=out_shape,
                 mode="constant",
                 cval=np.nan) for img, trfm in zip(img_list, trfm_list)
        ]

        all_nan_mask = np.all([np.isnan(img) for img in global_img_list],
                              axis=0)
        global_img_list[0][all_nan_mask] = 0.

        composite_img = np.nanmean(global_img_list, 0)
        composite_img = torch.from_numpy(composite_img)

        y, x = torch.nonzero(composite_img, as_tuple=True)

        t = y.min()
        l = x.min()
        b = y.max()
        r = x.max()

        channels.append(composite_img[t:b, l:r])

    merged_image = torch.stack((channels[0], channels[1], channels[2]), dim=0)
    return merged_image


def __compute_sift(image, x, y, radius):
    grayscale = ImageOps.grayscale(image)
    image_tensor = T.ToTensor()(grayscale)

    assert x.shape[0] == y.shape[0]

    circles = torch.tensor([])
    addresses: list[list[int]] = []

    for i in range(x.shape[0]):
        circle = torch.tensor([x[i], y[i], radius]).unsqueeze(0)
        circles = torch.cat((circles, circle))
        addresses.append([int(x[i]), int(y[i])])

    sift = sifTransform(image_tensor, circles, device, enlarge_factor=1)

    return sift, addresses


def __panorama2(image_left: Image.Image, image_right: Image.Image):
    l_corners, l_y, l_x = printRuntime(
        'Harris Left',
        lambda: __perform_harris(image_left, sigma=1.4, thresh=0.04, radius=4))

    plt.spy(l_corners, markersize=1)

    # plt.savefig('output/harris_left.png')

    plt.show()

    r_corners, r_y, r_x = printRuntime(
        'Harris Right', lambda: __perform_harris(
            image_right, sigma=1.4, thresh=0.04, radius=4))

    plt.spy(r_corners, markersize=1)

    # plt.savefig('output/harris_right.png')

    plt.show()

    sift_left, addr_left = printRuntime(
        'SIFT Left', lambda: __compute_sift(image_left, l_x, l_y, 16))
    sift_right, addr_right = printRuntime(
        'SIFT Right', lambda: __compute_sift(image_right, r_x, r_y, 16))

    distance = printRuntime(
        'Compute Distance',
        lambda: dist2(sift_left, sift_right),
    )

    pairs = printRuntime(
        'Find Matches',
        lambda: __find_matches(distance, 0.017),
    )

    pairs_length = pairs[0].shape[0]
    print(f'Pairs: {pairs_length}')

    plt.imshow(np.asarray(image_left), cmap="gray")

    for i in range(pairs_length):
        x, y = addr_left[pairs[1][i]]
        dist = 1 - distance[pairs[0][i], pairs[1][i]]
        marker_size = (10 * dist**40)
        plt.plot(x, y, marker="s", markersize=marker_size, fillstyle='none')

    # plt.savefig('output/pairs_left.png')
    plt.show()

    plt.imshow(np.asarray(image_right), cmap="gray")

    for i in range(pairs_length):
        x, y = addr_right[pairs[0][i]]
        dist = 1 - distance[pairs[0][i], pairs[1][i]]
        marker_size = (10 * dist**40)
        plt.plot(x, y, marker="s", markersize=marker_size, fillstyle='none')

    # plt.savefig('output/pairs_right.png')
    plt.show()

    r_output = printRuntime(
        'RANSAC',
        lambda: ransac(pairs, addr_left, addr_right, 4, 2000),
    )

    homography = r_output[0]
    inliers = r_output[1]
    inliers_length = inliers[0].shape[0]

    plt.imshow(np.asarray(image_left), cmap="gray")

    for i in range(inliers_length):
        x, y = addr_left[int(inliers[1][i])]
        dist = 1 - distance[int(inliers[0][i]), int(inliers[1][i])]
        marker_size = (10 * dist**40)
        plt.plot(x, y, marker="s", markersize=marker_size, fillstyle='none')

    # plt.savefig('output/inliers_left.png')
    plt.show()

    plt.imshow(np.asarray(image_right), cmap="gray")

    for i in range(inliers_length):
        x, y = addr_right[int(inliers[0][i])]
        dist = 1 - distance[int(inliers[0][i]), int(inliers[1][i])]
        marker_size = (10 * dist**40)
        plt.plot(x, y, marker="s", markersize=marker_size, fillstyle='none')

    # plt.savefig('output/inliers_right.png')
    plt.show()

    merged = printRuntime(
        'Merging Images', lambda: __merge_images(T.ToTensor()(image_left),
                                                 T.ToTensor()
                                                 (image_right), homography))

    merged_image = T.ToPILImage()(merged)

    return merged_image


def generate(images: list[Image.Image]):
    merged_image = images[0]
    for i in range(1, len(images)):
        merged_image = __panorama2(merged_image, images[i])

    return merged_image
