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