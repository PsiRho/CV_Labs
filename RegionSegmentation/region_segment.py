import cv2
import numpy as np
from Convolution.convolution import convolute
from Convolution.convolution import gaussian_kernel


def edge_detection_kernel(size: int = 3) -> np.ndarray:
    """
    Edge detection kernel based on the Laplacian of Gaussian.
    :param size: size of the kernel. e.g. 3, 5, 7, ...
    :return: the kernel to convolute with the image
    """
    # empty kernel
    kernel = np.zeros((size, size), dtype=np.float32)

    # set center to 1
    kernel[size // 2, size // 2] = 1

    # subtract gaussian kernel
    kernel = kernel - gaussian_kernel(size, 1, 1)
    print(kernel)  # TODO remove print

    return kernel


def region_segmentation(image: np.ndarray, kernel: np.ndarray, threshold: int = 0) -> np.ndarray:
    """
    Perform region segmentation on the given image.
    :param image: an image
    :param kernel: the kernel to convolute with the image
    :param threshold: the threshold to apply to the convoluted image
    :return: the segmented image
    """
    # convolute the image with the kernel
    convoluted_image = convolute(image, kernel, 1)

    # apply threshold
    segmented_image = np.where(convoluted_image > threshold, 255, 0)

    return segmented_image


def remove_isolated_pixels(image: np.ndarray, region_size: int) -> np.ndarray:
    """
    Remove isolated pixels from the image. A pixel is considered isolated if it has less than region_size white pixels
    in its 3x3 neighborhood.
    :param image: an image
    :param region_size: the minimum number of white pixels in the 3x3 neighborhood
    :return: the image with isolated pixels removed
    """
    # image dimensions
    i_row, i_col = image.shape[:2]

    # empty image for output
    output = np.zeros_like(image)

    # remove isolated pixels
    for y in range(i_row):
        for x in range(i_col):
            if image[y, x] == 255:
                # if pixel is not on the border
                if 0 < y < i_row - 1 and 0 < x < i_col - 1:
                    # if there are at least region_size white pixels in the 3x3 neighborhood
                    if (image[y - 1:y + 2, x - 1:x + 2] == 255).sum() >= region_size:
                        output[y, x] = 255

    return output

