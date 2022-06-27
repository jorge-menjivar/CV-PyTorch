[![Issues][issues]][issues-url]
[![PyPi][pypi]][license-url]
[![MIT License][license]][license-url]

<!-- [![Contributors][contributors]][contributors-url] -->
<!-- [![Forks][forks]][forks-url]s -->
<!-- [![Stargazers][stars]][stars-url] -->

# CV-PyTorch

## Installation

```sh
pip install cv-pytorch
```

## Use Cases

CV-PyTorch is a collection of tools to help facilitate computer vision operations.

Some of the tools in this library can be very useful to pre-process images before feeding to deep networks. However, the uses are not limited to pre-processing, you can perform a wide variety of operations that are helpful in many fields of work.

## Why?

The re-implementation of these tools in PyTorch has the advantage of being significantly faster than other libraries using NumPy, etc.

# Supported Operations

## Corner Detection (Harris)

<div>
    <img src="https://raw.githubusercontent.com/jorge-menjivar/CV-PyTorch/main/sample_images/uttower/harris_left.png" width="300"/>
    <img src="https://raw.githubusercontent.com/jorge-menjivar/CV-PyTorch/main/sample_images/uttower/harris_right.png" width="300"/>
</div>

## Pair Matching (SIFT)

<div>
    <img src="https://raw.githubusercontent.com/jorge-menjivar/CV-PyTorch/main/sample_images/uttower/pairs_left.png" width="300"/>
    <img src="https://raw.githubusercontent.com/jorge-menjivar/CV-PyTorch/main/sample_images/uttower/pairs_right.png" width="300"/>
</div>

## Inlier Detection (RANSAC)

<div>
    <img src="https://raw.githubusercontent.com/jorge-menjivar/CV-PyTorch/main/sample_images/uttower/inliers_left.png" width="300"/>
    <img src="https://raw.githubusercontent.com/jorge-menjivar/CV-PyTorch/main/sample_images/uttower/inliers_right.png" width="300"/>
</div>

## Pixel Matching

<img src="https://raw.githubusercontent.com/jorge-menjivar/CV-PyTorch/main/sample_images/uttower/merged_image.png" width="600"/>

## N Image Stitching

This is an example of 3 images stitched together:

<img src="https://raw.githubusercontent.com/jorge-menjivar/CV-PyTorch/main/sample_images/ledge.png" width="600"/>

## Gaussian Smoothing

<img src="https://raw.githubusercontent.com/jorge-menjivar/CV-PyTorch/main/sample_images/1.png" width="400"/>

## Sharpening

<img src="https://raw.githubusercontent.com/jorge-menjivar/CV-PyTorch/main/sample_images/4.png" width="400"/>

## Orientation

<img src="https://raw.githubusercontent.com/jorge-menjivar/CV-PyTorch/main/sample_images/5a.png" width="400"/>

## Magnitude

<img src="https://raw.githubusercontent.com/jorge-menjivar/CV-PyTorch/main/sample_images/5b.png" width="400"/>

## Scaling (No Interpolation)

<img src="https://raw.githubusercontent.com/jorge-menjivar/CV-PyTorch/main/sample_images/6a.png" width="400"/>

## Scaling (Bilinear Interpolation)

<img src="https://raw.githubusercontent.com/jorge-menjivar/CV-PyTorch/main/sample_images/6b.png" width="400"/>

## Edge Detection (Canny)

<div>
    <img src="https://raw.githubusercontent.com/jorge-menjivar/CV-PyTorch/main/sample_images/Circle.png" width="200"/>
    <img src="https://raw.githubusercontent.com/jorge-menjivar/CV-PyTorch/main/sample_images/7.png" width="200"/>
</div>

## Line Detection (Hough)

<img src="https://raw.githubusercontent.com/jorge-menjivar/CV-PyTorch/main/sample_images/t.png" width="400"/>

[contributors]: https://img.shields.io/github/contributors/jorge-menjivar/CV-PyTorch?style=for-the-badge
[contributors-url]: https://github.com/jorge-menjivar/CV-PyTorch/graphs/contributors
[forks]: https://img.shields.io/github/forks/jorge-menjivar/CV-PyTorch?style=for-the-badge
[forks-url]: https://github.com/jorge-menjivar/CV-PyTorch/network/members
[stars]: https://img.shields.io/github/stars/jorge-menjivar/CV-PyTorch?style=for-the-badge
[stars-url]: https://github.com/jorge-menjivar/CV-PyTorch/stargazers
[issues]: https://img.shields.io/github/issues/jorge-menjivar/CV-PyTorch?style=for-the-badge
[issues-url]: https://github.com/jorge-menjivar/CV-PyTorch/issues
[license]: https://img.shields.io/github/license/jorge-menjivar/CV-PyTorch?style=for-the-badge
[license-url]: https://github.com/jorge-menjivar/CV-PyTorch/blob/main/LICENSE
[pypi]: https://img.shields.io/pypi/v/CV-PyTorch?style=for-the-badge
[pypi-url]: https://pypi.org/project/CV-PyTorch/
