import cv2
import numpy as np


def convolve(img, kernel):
    """
    This function will convolve the input image with the input kernel.
    :param img: The input image.
    :param kernel: The input kernel.
    :return: The convoluted image.
    """
    # image dimensions
    img_height = img.shape[0]
    img_width = img.shape[1]

    # kernel dimensions
    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]

    # padding
    pad_height = int((kernel_height - 1) / 2)
    pad_width = int((kernel_width - 1) / 2)

    # Create the padded image
    padded_img = np.zeros((img_height + (2 * pad_height), img_width + (2 * pad_width)), dtype=np.uint8)
    padded_img[pad_height:padded_img.shape[0] - pad_height, pad_width:padded_img.shape[1] - pad_width] = img

    # Create the convoluted image
    convoluted_img = np.zeros((img_height, img_width), dtype=np.uint8)

    # Convolute the image
    for i in range(img_height):
        for j in range(img_width):
            convoluted_img[i, j] = np.sum(padded_img[i:i + kernel_height, j:j + kernel_width] * kernel)

    return convoluted_img


# Read the image
test_img = cv2.imread('Res/flower.jpg', 0)
cv2.imshow('Original', test_img)

# Create the kernel
gaussian_kernel_3x3 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 1 / 16
gaussian_kernel_5x5 = np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4],
                                [1, 4, 6, 4, 1]], dtype=np.float32) / 1 / 256


def thresholding(img, threshold):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] <= threshold:
                img[i][j] = 0
    return img


#thresholded = thresholding(test_img, 82)
blurred5x5 = convolve(test_img, gaussian_kernel_5x5)
blurred3x3 = convolve(test_img, gaussian_kernel_3x3)


# Show the images
cv2.imshow('Convoluted3x3', blurred3x3)
cv2.imshow('Convoluted5x5', blurred5x5)

# press any key to close all windows made by cv2
cv2.waitKey(0)
cv2.destroyAllWindows()
