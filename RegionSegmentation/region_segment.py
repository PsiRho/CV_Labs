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