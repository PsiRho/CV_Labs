import time
import timeit

import cv2
import numpy as np
import Convolution.convolution as conv


def edge_detection_kernel(size: int = 3, sigma: float = 1) -> np.ndarray:
    """
    Edge detection kernel based on the Laplacian of Gaussian.
    :param size: size of the kernel. e.g. 3, 5, 7, ... If size is even, it will be incremented by 1. If size is less
    than 3, it will be set to 3.
    :param sigma: SD of the Gaussian distribution
    :return: the kernel to convolute with the image
    """
    # set size
    size = max(3, size + 1 if size % 2 == 0 else size)

    # empty kernel
    kernel = np.zeros((size, size), dtype=np.float32)

    # set center to 1
    kernel[size // 2, size // 2] = 1

    # subtract gaussian kernel
    kernel = kernel - conv.gaussian_kernel(sigma, size=size, channels=1)

    return kernel


def laplacian_of_gaussian(image: np.ndarray, kernel: np.ndarray, threshold: int = 150) -> np.ndarray:
    """
    Perform region segmentation on the given image.
    :param image: an image
    :param kernel: the kernel to convolute with the image
    :param threshold: the threshold to apply to the convoluted image
    :return: the segmented image
    """
    # convolute the image with the kernel
    convoluted_image = conv.convolute(image, kernel, 1)

    # apply threshold
    #segmented_image = np.where(convoluted_image > threshold, 255, 0)

    return convoluted_image


def difference_of_gaussian(image: np.ndarray, kernel1: np.ndarray, kernel2: np.ndarray) -> np.ndarray:
    """
    Perform difference of Gaussians on the given image.
    :param image: an image
    :param kernel1: the first kernel to convolute with the image
    :param kernel2: the second kernel to convolute with the image
    :return: the segmented image
    """

    if kernel1.shape == kernel2.shape:
        dog_kernel = kernel1 - kernel2
        diff_of_gaussian = conv.convolute(image, dog_kernel, 1)
    else:
        convoluted_image1 = conv.convolute(image, kernel1, 1)
        convoluted_image2 = conv.convolute(image, kernel2, 1)
        diff_of_gaussian = convoluted_image1 - convoluted_image2

    return diff_of_gaussian


def remove_isolated_pixels(image: np.ndarray, region_size: int = 3) -> np.ndarray:
    """
    Remove isolated pixels from the image. A pixel is considered isolated if it has less than min_region_size white
    pixels in a neighbourhood.
    :param image: an image
    :param region_size: the minimum number of white pixels in the neighbourhood
    :return: the image with isolated pixels removed
    """
    # get image dimensions
    row, col = image.shape[:2]

    # create output image
    output = np.zeros_like(image)

    # remove isolated pixels
    for y in range(row):
        for x in range(col):
            if image[y, x] == 255:
                # if pixel is not on the border or has at least region_size white pixels in its 3x3 neighborhood
                if (y == 0 or y == row - 1 or x == 0 or x == col - 1) or (
                        image[y - 1:y + 2, x - 1:x + 2] == 255).sum() >= region_size:
                    output[y, x] = 255

    return output


def colorize_edges(orig_img_color, segmented_img_grayscale):
    """
    Colorize the edges of the segmented image. Will take the colors from the original image and apply them to the
    segmented image where there are edges (white pixels).
    :param orig_img_color: the original image
    :param segmented_img_grayscale: the segmented image
    :return: the colorized image
    """
    # get image dimensions
    row, col = orig_img_color.shape[:2]

    # create output image
    output = np.zeros_like(orig_img_color)

    # apply color to segmented image
    for y in range(row):
        for x in range(col):
            if segmented_img_grayscale[y, x] == 255:
                output[y, x] = orig_img_color[y, x]

    return output


# Wrapper for laplacian of gaussian / marr hildreth edge detection
def do_log(image_path: str, kernel_size: int = 3, sigma: float = 1, region_size: int = 7,
           clean_image: bool = False, colorize: bool = False):
    # read image
    orig_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    orig_img_gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)

    # create edge detection kernel
    edge_detect_kern = edge_detection_kernel(kernel_size, sigma)

    # perform segmentation
    segmented_img = laplacian_of_gaussian(orig_img_gray, edge_detect_kern)

    # remove unwanted regions
    if clean_image:
        segmented_img = remove_isolated_pixels(segmented_img, region_size)

    # colorize edges
    if colorize:
        segmented_img = colorize_edges(orig_img, segmented_img)

    # convert images to uint8
    segmented_img = np.uint8(segmented_img)

    # get image name
    img_name = "{0}_log_threshold{1}{2}".format(image_path.split('/')[-1].split('.')[0],
                                                ('_colored' if colorize else ''),
                                                ('_cleaned_with_regsize' + str(region_size) if clean_image else ''))

    # display image
    cv2.imshow(img_name, segmented_img)

    return segmented_img


# wrapper for difference of gaussians edge detection
def do_dog(image_path: str, kernel_size: int = 5, sigma: float = -1, dog_sigma_ratio: float = 1.6, region_size: int = 1,
           clean_image: bool = False, colorize: bool = False):
    # read image
    orig_img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # convert to grayscale
    orig_img_gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)

    # calculate sigma for the second kernel, not an elegant solution as it is duplicated in the gaussian kernel function
    sigma2 = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8

    # create kernels
    kernel1 = conv.gaussian_kernel(kernel_size, sigma, channels=1)
    kernel2 = conv.gaussian_kernel(kernel_size, sigma2 * dog_sigma_ratio, channels=1)

    # perform segmentation
    segmented_img = difference_of_gaussian(orig_img_gray, kernel1, kernel2)

    # remove unwanted regions
    if clean_image:
        segmented_img = remove_isolated_pixels(segmented_img, region_size)

    # colorize edges
    if colorize:
        segmented_img = colorize_edges(orig_img, segmented_img)

    # convert to uint8
    segmented_img = np.uint8(segmented_img)

    # get image name
    img_name = "{0}_dog_threshold{1}{2}".format(image_path.split('/')[-1].split('.')[0],
                                                   ('_colored' if colorize else ''),
                                                   ('_cleaned_with_regsize' + str(region_size) if clean_image else ''))

    # display image
    cv2.imshow(img_name, segmented_img)

    return segmented_img


def main():
    # do_log('../Res/flower.jpg', sigma=4, threshold=50, region_size=3,
    #      clean_image=True, colorize=True)

    do_dog('../Res/flower.jpg', kernel_size=5, dog_sigma_ratio=1.6, region_size=1, clean_image=True)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
