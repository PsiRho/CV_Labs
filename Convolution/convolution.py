import numpy as np
from numpy import ndarray
from Util.padding import padding


def gaussian_kernel(size: int = 5, sigma: float = -1, channels: int = 1) -> ndarray:
    """
    Create a Gaussian kernel with the given sigma. The formula used is:
    G(x, y) = 1 / (2 * pi * sigma^2) * e^(-(x^2 + y^2) / (2 * sigma^2))
    Could have used kernel_size = 2 * int(4 * sigma + 0.5) + 1 and sigma = (kernel_size - 1) / (2 * 2.575)
    Can use either of these to calculate the kernel size or the sigma according to the internet.
    :param size: The size of the kernel. If the size is less than 3, it will be set to 3. If the size is even, it will
    be incremented by 1.
    :param sigma: The standard deviation of the Gaussian distribution. If sigma is -1, it will be calculated using the
    formula: sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8.
    :param channels: The number of channels in the image. Default is 1.
    :return: The Gaussian kernel.
    """

    # make sure size is odd and at least 3
    size = max(3, size + 1 if size % 2 == 0 else size)

    # calculate sigma if not given
    if sigma == -1:
        sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8

    center = size // 2
    x, y = np.mgrid[-center:center + 1, -center:center + 1]  # x, y grid
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))  # calculate the Gaussian distribution
    kernel /= np.sum(kernel)  # normalize sum to 1
    if channels > 1:
        kernel = np.stack([kernel] * channels, axis=-1)
    return kernel


def convolute(image, kernel, channels: int):
    # get image dimensions
    row, col = image.shape[:2]
    chan = channels

    # get kernel dimensions
    k_row, k_col = kernel.shape[:2]

    # create output image
    output = np.zeros_like(image)

    # add zero padding
    padded_image = padding(image, kernel, channels)

    # convolute
    for y in range(row):
        for x in range(col):
            if chan == 1:
                output[y, x] = np.sum(kernel * padded_image[y: y + k_row, x: x + k_col])
            else:
                for c in range(chan):
                    output[y, x, c] = np.sum(kernel[:, :, c] * padded_image[y: y + k_row, x: x + k_col, c])

    return output


