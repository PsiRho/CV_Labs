import cv2
import numpy as np


def gaussian_kernel(size: int, sigma: float, channels: int) -> np.ndarray:
    """
    Create a Gaussian kernel with the given size and sigma.
    :param size: The size of the kernel.
    :param sigma: The standard deviation of the Gaussian distribution.
    :param channels: The number of channels in the image. Default is 1.
    :return: The Gaussian kernel.
    """

    center = size // 2
    x, y = np.mgrid[-center:center + 1, -center:center + 1]
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)
    if channels > 1:
        kernel = np.stack([kernel] * channels, axis=-1)
    return kernel


def convolute(image, kernel, num_channels):
    # get image dimensions
    i_row, i_col = image.shape[:2]
    i_chan = num_channels

    # get kernel dimensions
    k_row, k_col = kernel.shape[:2]
    k_chan = num_channels

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


# read image
orig_img_flower = cv2.imread('Res/flower.jpg', cv2.IMREAD_COLOR)
orig_img_tiger = cv2.imread('Res/tiger.jpg', cv2.IMREAD_GRAYSCALE)

# create gaussian kernel
gaus_kernel = gaussian_kernel(3, 1, 3)
gaus_kernel_tiger = gaussian_kernel(3, 1, 1)

# convolute image
gaussian = convolute(orig_img_flower, gaus_kernel, 3)
gaussian_tiger = convolute(orig_img_tiger, gaus_kernel_tiger, 1)

# show images
cv2.imshow('Original_flower', orig_img_flower)
cv2.imshow('Convoluted_flower', gaussian)
cv2.imshow('Original_tiger', orig_img_tiger)
cv2.imshow('Convoluted_tiger', gaussian_tiger)

# wait for key press
cv2.waitKey(0)
cv2.destroyAllWindows()
