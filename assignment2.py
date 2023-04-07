import random
import string

import cv2
import numpy as np
from numpy import ndarray
from scipy.spatial.distance import cdist

PATH_FLOWER = 'Res/flower.jpg'
PATH_TIGER = 'Res/tiger.jpg'


# ################################################### Edge detection ###################################################

def difference_of_gaussian(image: np.ndarray, kernel_small_sigma: np.ndarray, kernel_large_sigma: np.ndarray,
                           threshold: int = 150) -> np.ndarray:
    # image with low SD
    convoluted_small_sigma = convolute(image, kernel_small_sigma)

    # image with high SD
    convoluted_large_sigma = convolute(image, kernel_large_sigma)

    # difference of the two images
    diff_of_gaussian = convoluted_large_sigma - convoluted_small_sigma

    # threshold
    diff_of_gaussian = np.where(diff_of_gaussian >= threshold, 255, 0)

    return diff_of_gaussian.astype(np.uint8)


def laplacian_of_gaussian(image: np.ndarray, size: int = 3, sigma: float = -1, threshold: int = 150) -> np.ndarray:
    # odd size and at least 3
    size = max(3, size + 1 if size % 2 == 0 else size)

    # empty kernel
    kernel = np.zeros((size, size), dtype=np.float32)

    # center pixel is 1
    kernel[size // 2, size // 2] = 1

    # subtract Gaussian kernel
    kernel = kernel - gaussian_kernel(size, sigma, channels=1)

    # convolute
    convoluted_image = convolute(image, kernel)

    # threshold
    convoluted_image = np.where(convoluted_image >= threshold, 255, 0)

    return convoluted_image


def gaussian_kernel(size: int = 5, sigma: float = -1, channels: int = 1) -> ndarray:
    # odd and at least 3
    size = max(3, size + 1 if size % 2 == 0 else size)

    # calculate sigma if not given
    if sigma == -1:
        sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8

    # center pixel
    center = size // 2

    # x, y grid
    x, y = np.mgrid[-center:center + 1, -center:center + 1]

    # calculate the Gaussian distribution
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    # normalize sum to 1
    kernel /= np.sum(kernel)

    # stack channels
    if channels > 1:
        kernel = np.stack([kernel] * channels, axis=-1)
    return kernel


# ###################################################### K-means #######################################################


def random_pixel(img, k: int) -> np.ndarray:
    # flatten image
    flat_img = img.reshape(-1, img.shape[-1])

    # random pixels
    return img[np.random.choice(flat_img.shape[0], k, replace=False), :]


def initialize_clusters(img, k: int) -> np.ndarray:
    # cluster centers with k rows and 3 cols for channels
    cluster_centers = np.zeros((k, 3))

    # first center is random pixel
    cluster_centers[0] = random_pixel(img, 1)

    # loop over clusters
    for i in range(1, k):
        # distances from previous centers
        distances = cdist(img, np.array([cluster_centers[i - 1]]))[:, 0]

        # pixel farthest from previous centers
        cluster_centers[i] = img[np.argmax(distances), :]

    return cluster_centers


def update_clusters(img, labels, k: int) -> np.ndarray:
    # cluster centers with k rows and 3 cols for channels
    new_cluster_centers = np.zeros((k, 3))

    # loop over clusters
    for i in range(k):

        # if cluster is not empty
        if np.sum(labels == i) != 0:

            # update to mean of pixels in cluster
            new_cluster_centers[i] = np.mean(img[labels == i], axis=0)

        else:  # if cluster is empty
            # set center to random pixel
            new_cluster_centers[i] = random_pixel(img, 1)

    return new_cluster_centers


def assign_labels(img, cluster_centers) -> np.ndarray:
    # assign pixels to closest cluster center
    distances = cdist(img, cluster_centers)

    # return cluster index
    return np.argmin(distances, axis=1)


def k_means_clustering(image, k: int, tolerance: float) -> (np.ndarray, np.ndarray):
    # convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # image dims
    height, width = image.shape[:2]

    # flatten image
    img = image.reshape((height * width, 3))

    # initialize clusters
    cluster_centers = initialize_clusters(img, k)

    # iteration counter
    iterations = 0

    # loop until convergence
    while True:
        iterations += 1

        # assign labels
        labels = assign_labels(img, cluster_centers)

        # update clusters
        new_cluster_centers = update_clusters(img, labels, k)

        # break if converged. converged if cluster centers don't change more than tolerance
        if np.allclose(cluster_centers, new_cluster_centers, rtol=tolerance):
            break

        # update cluster centers
        cluster_centers = new_cluster_centers

    # print iteration counter and number of clusters
    print(f'k-means clustering converged after {iterations} iterations with {k} clusters.')

    # reshape labels
    labels_2d = labels.reshape((height, width))

    return labels_2d, cluster_centers


# ######################################################## Utils #######################################################


def create_result_img(labels, cluster_centers) -> np.ndarray:
    # image dims
    height, width = labels.shape[:2]

    # create image
    image = cluster_centers[labels.flatten()].reshape((height, width, -1))

    # convert to uint8
    image = image.astype(np.uint8)

    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    # set size
    kernel_size = max(3, kernel_size + 1 if kernel_size % 2 == 0 else kernel_size)

    # pad image
    padded_image = pad(image, np.ones((kernel_size, kernel_size), dtype=np.float32))

    # apply filter
    result_img = np.zeros_like(image)
    for i, j in np.ndindex(image.shape[:2]):
        patch = padded_image[i:i + kernel_size, j:j + kernel_size]
        result_img[i, j] = np.median(patch, axis=(0, 1))

    return result_img.astype(image.dtype)


def pad(image, kernel, pad_type='constant'):
    # padding dims
    pad_height = kernel.shape[0] // 2
    pad_width = kernel.shape[1] // 2

    # pad image
    if len(image.shape) == 2:  # grayscale
        padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), pad_type, constant_values=0)
    else:  # color
        padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), pad_type,
                              constant_values=0)

    return padded_image


def convolute(image, kernel):
    # get image dimensions
    row, col = image.shape[:2]
    if len(image.shape) == 2:
        chan = 1
    else:
        chan = image.shape[2]

    # get kernel dimensions
    k_row, k_col = kernel.shape[:2]

    # create output image
    output = np.zeros_like(image)

    # add zero padding
    padded_image = pad(image, kernel)

    # loop over image
    for c in range(chan):
        for y in range(row):
            for x in range(col):
                if chan == 1:  # grayscale
                    # element-wise multiplication of the kernel and the image
                    output[y, x] = np.sum(kernel * padded_image[y: y + k_row, x: x + k_col])
                else:  # color
                    output[y, x, c] = np.sum(kernel * padded_image[y: y + k_row, x: x + k_col, c])

    return output


def colorize_edges(orig_img_color, segmented_img_grayscale):
    # boolean mask for white pixels
    white_pixels = segmented_img_grayscale == 255

    # colorize edges
    output = np.where(white_pixels[..., None], orig_img_color, 0)

    return output


def showcase_dog(kernel_size: int = 5, sigma: float = -1, sigma_ratio: float = 1.5, colorize: bool = False):
    # read image
    original_img_flower = cv2.imread(PATH_FLOWER, cv2.IMREAD_COLOR)
    original_img_tiger = cv2.imread(PATH_TIGER, cv2.IMREAD_COLOR)

    # convert to grayscale
    gray_img_flower = cv2.cvtColor(original_img_flower, cv2.COLOR_BGR2GRAY)
    gray_img_tiger = cv2.cvtColor(original_img_tiger, cv2.COLOR_BGR2GRAY)

    # create kernels
    if sigma == -1:
        sigma = (0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8)
    kernel1 = gaussian_kernel(size=kernel_size, sigma=sigma)
    kernel2 = gaussian_kernel(size=kernel_size, sigma=sigma * sigma_ratio)

    # DoG
    dog_img_flower = difference_of_gaussian(gray_img_flower, kernel1, kernel2)
    dog_img_tiger = difference_of_gaussian(gray_img_tiger, kernel1, kernel2)

    # colorize edges
    if colorize:
        dog_img_flower = colorize_edges(original_img_flower, dog_img_flower)
        dog_img_tiger = colorize_edges(original_img_tiger, dog_img_tiger)

    # show image
    cv2.imshow(''.join(random.choices(string.ascii_uppercase + string.digits, k=7)), dog_img_flower)
    cv2.imshow(''.join(random.choices(string.ascii_uppercase + string.digits, k=7)), dog_img_tiger)


def showcase_kmeans(k: int = 2, tolerance: float = 1e-4):
    # read image
    original_img_flower = cv2.imread(PATH_FLOWER, cv2.IMREAD_COLOR)
    original_img_tiger = cv2.imread(PATH_TIGER, cv2.IMREAD_COLOR)

    # k-means
    labels_flower, cluster_centers_flower = k_means_clustering(original_img_flower, k, tolerance=tolerance)
    labels_tiger, cluster_centers_tiger = k_means_clustering(original_img_tiger, k, tolerance=tolerance)

    # create result image
    result_img_flower = create_result_img(labels_flower, cluster_centers_flower)
    result_img_tiger = create_result_img(labels_tiger, cluster_centers_tiger)

    # show image
    cv2.imshow(f'flower_k={k}_' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=5)),
               result_img_flower)
    cv2.imshow(f'tiger_k={k}_' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=5)), result_img_tiger)


def showcase_median_filter_dog(gaussian_kernel_size: int = 5, median_blur_kernel_size: int = 7, sigma: float = -1,
                               sigma_ratio: float = 1.5, colorize: bool = False):
    # read images
    original_img_flower = cv2.imread(PATH_FLOWER, cv2.IMREAD_COLOR)
    original_img_tiger = cv2.imread(PATH_TIGER, cv2.IMREAD_COLOR)

    # convert to grayscale
    gray_img_flower = cv2.cvtColor(original_img_flower, cv2.COLOR_BGR2GRAY)
    gray_img_tiger = cv2.cvtColor(original_img_tiger, cv2.COLOR_BGR2GRAY)

    # median filter
    median_img_flower = median_filter(gray_img_flower, median_blur_kernel_size)
    median_img_tiger = median_filter(gray_img_tiger, median_blur_kernel_size)

    # create kernels
    if sigma == -1:
        sigma = (0.3 * ((gaussian_kernel_size - 1) * 0.5 - 1) + 0.8)
    small_sigma = gaussian_kernel(size=gaussian_kernel_size, sigma=sigma)
    large_sigma = gaussian_kernel(size=gaussian_kernel_size, sigma=sigma * sigma_ratio)

    # DoG
    dog_img_flower = difference_of_gaussian(median_img_flower, kernel_small_sigma=small_sigma,
                                            kernel_large_sigma=large_sigma)
    dog_img_tiger = difference_of_gaussian(median_img_tiger, kernel_small_sigma=small_sigma,
                                           kernel_large_sigma=large_sigma)

    # colorize edges
    if colorize:
        dog_img_flower = colorize_edges(original_img_flower, dog_img_flower)
        dog_img_tiger = colorize_edges(original_img_tiger, dog_img_tiger)

    # show image
    cv2.imshow(''.join(random.choices(string.ascii_uppercase + string.digits, k=5)), dog_img_flower)
    cv2.imshow(''.join(random.choices(string.ascii_uppercase + string.digits, k=5)), dog_img_tiger)


def showcase_median_filter_kmeans(median_blur_kernel_size: int = 7, k: int = 4, tolerance: float = 1e-4):
    # read image
    original_img_flower = cv2.imread(PATH_FLOWER, cv2.IMREAD_COLOR)
    original_img_tiger = cv2.imread(PATH_TIGER, cv2.IMREAD_COLOR)

    # median filter
    median_img_flower = median_filter(original_img_flower, median_blur_kernel_size)
    median_img_tiger = median_filter(original_img_tiger, median_blur_kernel_size)

    # k-means
    labels_flower, cluster_centers_flower = k_means_clustering(median_img_flower, k, tolerance=tolerance)
    labels_tiger, cluster_centers_tiger = k_means_clustering(median_img_tiger, k, tolerance=tolerance)

    # create result image
    result_img_flower = create_result_img(labels_flower, cluster_centers_flower)
    result_img_tiger = create_result_img(labels_tiger, cluster_centers_tiger)

    # show image
    cv2.imshow(f'flower_k={k}_' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=5)),
               result_img_flower)
    cv2.imshow(f'tiger_k={k}_' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=5)), result_img_tiger)


# ######################################################## Main ########################################################


def main(*args):

    # showcase the difference of gaussian
    if args[0] == 1:
        showcase_dog(kernel_size=5, sigma=-1, sigma_ratio=1.6, colorize=False)
        showcase_dog(kernel_size=5, sigma=-1, sigma_ratio=1.6, colorize=True)

    # showcase the median filter and the difference of gaussian
    elif args[0] == 2:
        showcase_median_filter_dog(gaussian_kernel_size=5, median_blur_kernel_size=5, sigma=-1, sigma_ratio=1.6,
                                   colorize=False)
        showcase_median_filter_dog(gaussian_kernel_size=5, median_blur_kernel_size=5, sigma=-1, sigma_ratio=1.6,
                                   colorize=True)

    # showcase the k-means with different k values
    elif args[0] == 3:
        # loop over different k values
        for i in range(2, 5):
            showcase_kmeans(k=i, tolerance=1e-4)

    # showcase the median filter and the k-means
    elif args[0] == 4:
        showcase_median_filter_kmeans(median_blur_kernel_size=7, k=4, tolerance=1e-4)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':

    # The code is not optimized for performance, so it can take a while to run (15-30 sec on my computer).

    # main(1) = showcase the difference of gaussian
    # main(2) = showcase the median filter and the difference of gaussian
    # main(3) = showcase the k-means with different k values
    # main(4) = showcase the median filter and the k-means

    main(1)
