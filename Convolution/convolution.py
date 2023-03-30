import numpy as np
from numpy import ndarray


def gaussian_kernel(size: int = 5, sigma: float = -1, channels: int = 1) -> ndarray:
    """
    Create a Gaussian kernel with the given sigma. The formula used is:
    G(x, y) = 1 / (2 * pi * sigma^2) * e^(-(x^2 + y^2) / (2 * sigma^2))
    Could have used kernel_size = 2 * int(4 * sigma + 0.5) + 1 and sigma = (kernel_size - 1) / (2 * 2.575)
    Either of these to calculate the kernel size or the sigma.
    :param size: The size of the kernel. If size is -1, the size will be 6 * sigma to ensure that the kernel is large
    enough to capture the Gaussian distribution. If size is even, it will be incremented by 1. If size is less than 3,
    it will be set to 3.
    :param sigma: The standard deviation of the Gaussian distribution.
    :param channels: The number of channels in the image. Default is 1.
    :return: The Gaussian kernel.
    """

    # make sure size is odd and at least 3
    size = max(3, size + 1 if size % 2 == 0 else size)

    # calculate sigma if not given
    if sigma == -1:
        sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8

    center = size // 2
    x, y = np.mgrid[-center:center + 1, -center:center + 1]
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)
    if channels > 1:
        kernel = np.stack([kernel] * channels, axis=-1)
    return kernel


def convolute(image, kernel, num_channels: int):
    # get image dimensions
    i_row, i_col = image.shape[:2]
    i_chan = num_channels

    # get kernel dimensions
    k_row, k_col = kernel.shape[:2]

    # create output image
    output = np.zeros_like(image)

    # add zero padding
    pad_width = k_row // 2
    if num_channels == 1:
        padding = ((pad_width, pad_width), (pad_width, pad_width))
    else:
        padding = ((pad_width, pad_width), (pad_width, pad_width), (0, 0))
    padded_image = np.pad(image, padding, 'constant', constant_values=0)

    # convolute
    for y in range(i_row):
        for x in range(i_col):
            if i_chan == 1:
                output[y, x] = (kernel * padded_image[y: y + k_row, x: x + k_col]).sum()
            else:
                for c in range(i_chan):
                    output[y, x, c] = (kernel[:, :, c] * padded_image[y: y + k_row, x: x + k_col, c]).sum()

    return output

# # read image
# orig_img_color = cv2.imread('Res/flower.jpg', cv2.IMREAD_COLOR)
# orig_img_grayscale = cv2.imread('Res/tiger.jpg', cv2.IMREAD_GRAYSCALE)
#
# # create gaussian kernel
# gaussian_kernel_color = gaussian_kernel(3, 1, 3)
# gaussian_kernel_grayscale = gaussian_kernel(3, 1, 1)
#
# # convolute image
# conv_img_color = convolute(orig_img_color, gaussian_kernel_color, 3)
# conv_img_grayscale = convolute(orig_img_grayscale, gaussian_kernel_grayscale, 1)
#
# # show images
# cv2.imshow('Original_flower', orig_img_color)
# cv2.imshow('Convoluted_flower', conv_img_color)
# cv2.imshow('Original_tiger', orig_img_grayscale)
# cv2.imshow('Convoluted_tiger', conv_img_grayscale)
#
# # wait for key press
# cv2.waitKey(0)
# cv2.destroyAllWindows()
