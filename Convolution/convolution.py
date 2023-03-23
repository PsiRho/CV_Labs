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
    kernel = np.zeros((size, size, channels), dtype=np.float32)
    center = size // 2
    x, y = np.mgrid[-center:center + 1, -center:center + 1]
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)
    if channels > 1:
        kernel = np.stack([kernel] * channels, axis=-1)
    return kernel


def convolute2(img, kernel):
    """
    This function will convolute the input image with the input kernel.
    :param img: The input image.
    :param kernel: The input kernel.
    :return: The convoluted image.
    """
    # kernel dimensions
    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]

    # padding
    pad_height = int((kernel_height - 1) / 2)
    pad_width = int((kernel_width - 1) / 2)

    # create padded image
    padded_img = np.pad(img, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='constant')

    # create convoluted image
    convoluted_img = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=np.uint8)

    # perform convolution
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                convoluted_img[i, j, k] = np.sum(
                    padded_img[i:i + kernel_height, j:j + kernel_width, k] * kernel[:, :, k])

    return convoluted_img


# read image
orig_img = cv2.imread('../Res/flower.jpg', cv2.IMREAD_COLOR)

# create gaussian kernel
gaus_kernel = gaussian_kernel(9, 3, 3)

# convolute image
gaussian = convolute2(orig_img, gaus_kernel)

# show images
cv2.imshow('Original', orig_img)
cv2.imshow('Convoluted', gaussian)

# wait for key press
cv2.waitKey(0)
cv2.destroyAllWindows()
