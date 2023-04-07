import cv2
import numpy as np
import matplotlib

import Convolution.convolution as conv
from matplotlib import pyplot as plt
import MorphologicalOps.morphop as mrph
import Util.colorize as clr

matplotlib.use('Qt5Agg')


def laplacian_of_gaussian(image: np.ndarray, size: int = 3, sigma: float = -1, threshold: int = 150) -> np.ndarray:
    """
    Edge detection kernel based on the Laplacian of Gaussian.
    :param image: an image
    :param threshold: threshold for the edge detection
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
    kernel = kernel - conv.gaussian_kernel(size, sigma, channels=1)

    # convolute
    convoluted_image = conv.convolute(image, kernel)

    # threshold
    convoluted_image = np.where(convoluted_image >= threshold, 255, 0)

    return convoluted_image


def difference_of_gaussian(image: np.ndarray, kernel1: np.ndarray, kernel2: np.ndarray,
                           threshold: int = 150) -> np.ndarray:
    """
    Perform difference of Gaussians on the given image.
    :param image: an image
    :param threshold: threshold for the edge detection
    :param kernel1: the first kernel to convolute with the image
    :param kernel2: the second kernel to convolute with the image
    :return: the segmented image
    """
    convoluted_image1 = conv.convolute(image, kernel1)  # Image with low SD
    convoluted_image2 = conv.convolute(image, kernel2)  # Image with high SD
    diff_of_gaussian = convoluted_image2 - convoluted_image1

    # threshold
    diff_of_gaussian = np.where(diff_of_gaussian >= threshold, 255, 0)

    return diff_of_gaussian


def mean_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply a mean filter to the image. This will work for both grayscale and color images.
    :param image: an image
    :param kernel_size: size of the kernel. e.g. 3, 5, 7, ... If size is even, it will be incremented by 1. If size is less
    than 3, it will be set to 3.
    :return: the filtered image
    """
    # set size
    kernel_size = max(3, kernel_size + 1 if kernel_size % 2 == 0 else kernel_size)

    # create kernel
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)

    # convolute
    convoluted_image = conv.convolute(image, kernel)

    return convoluted_image


def median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply a median filter to the image. This will work for both grayscale and color images.
    :param image: an image
    :param kernel_size: size of the kernel. e.g. 3, 5, 7, ... If size is even, it will be incremented by 1. If size is less
    than 3, it will be set to 3.
    :return: the filtered image
    """
    # set size
    kernel_size = max(3, kernel_size + 1 if kernel_size % 2 == 0 else kernel_size)

    # convolute
    convoluted_image = np.zeros_like(image, dtype=np.float32)
    border_size = kernel_size // 2
    padded_image = np.pad(image, ((border_size, border_size), (border_size, border_size), (0, 0)), mode='reflect')

    for y in range(border_size, padded_image.shape[0] - border_size):
        for x in range(border_size, padded_image.shape[1] - border_size):
            window = padded_image[y - border_size:y + border_size + 1, x - border_size:x + border_size + 1]
            convoluted_image[y - border_size, x - border_size] = np.median(window, axis=(0, 1))

    return convoluted_image.astype(image.dtype)


def action(*args, **kwargs):
    """
    Wrapper function for performing the segmentation. This is a bit of a mess, but it makes testing different params
    easier.
    TODO: This function should REALLY be refactored.
    :param args: image path, DoG (Difference of gaussian) or LoG (Laplacian of Gaussian)
    :param kwargs: kernel_size: int, sigma: float, DoG_sigma_ratio, threshold: int,
    morphological_operation_type ['erode', 'dilate', 'opening', 'closing'], colorize: bool.
    operation kernel size
    :return: the segmented image
    """
    # if the first arg is a path, read the image and if its an image use that
    if isinstance(args[0], str):
        orig_img = cv2.imread(args[0], cv2.IMREAD_COLOR)
    else:
        orig_img = args[0]

    # convert to grayscale
    orig_img_gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)

    # get kernel size
    kernel_size = kwargs.get('kernel_size', 5)

    if str(args[1].casefold()) == 'DoG'.casefold():

        # create kernels
        if kwargs.get('sigma', -1) != -1:
            kernel1 = conv.gaussian_kernel(kernel_size, kwargs.get('sigma', -1), channels=1)
            kernel2 = conv.gaussian_kernel(kernel_size, kwargs.get('sigma', -1) * kwargs.get('dog_sigma_ratio', 1.6),
                                           channels=1)
        else:
            kernel1 = conv.gaussian_kernel(kernel_size, -1, channels=1)
            kernel2 = conv.gaussian_kernel(kernel_size,
                                           (0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8) * kwargs.get('dog_sigma_ratio',
                                                                                                    1.6),
                                           channels=1)

        # perform segmentation
        image = difference_of_gaussian(orig_img_gray, kernel1, kernel2, kwargs.get('threshold', 150))
    else:
        image = laplacian_of_gaussian(orig_img_gray, kernel_size, kwargs.get('sigma', -1), kwargs.get('threshold', 150))

    # morphological operation
    morph_op = kwargs.get('morph_op', '').lower()
    if morph_op != '' and morph_op in ['erode', 'dilate', 'opening', 'closing']:
        if morph_op in ['erode', 'dilate']:
            image = mrph.morph_op(image, morph_op, kwargs.get('morph_kernel_size', 3))
        elif morph_op == 'opening':
            image = mrph.morph_op(image, 'erode', kwargs.get('morph_kernel_size', 3))
            image = mrph.morph_op(image, 'dilate', kwargs.get('morph_kernel_size', 3))
        elif morph_op == 'closing':
            image = mrph.morph_op(image, 'dilate', kwargs.get('morph_kernel_size', 3))
            image = mrph.morph_op(image, 'erode', kwargs.get('morph_kernel_size', 3))

    # colorize edges
    if kwargs.get('colorize', False):
        image = clr.colorize_edges(orig_img, image)

    # convert to uint8
    image = np.uint8(image)

    # img_name = "{0}{1}{2}{3}{4}{5}{6}{7}".format(args[0].split('/')[-1].split('.')[0],
    #                                             '_' + args[1],
    #                                             '_kernelSize' + str(kernel_size),
    #                                             '_sigma' + str(kwargs.get('sigma', -1))
    #                                             if kwargs.get('sigma', -1) != -1
    #                                             else '_computedSigma',
    #                                             '_threshold' + str(kwargs.get('threshold', 150)),
    #                                             '_sigmaRatio' + str(kwargs.get('dog_sigma_ratio', 1.6)),
    #                                             ('_colored' if kwargs.get('colorize', False) else ''),
    #                                             ('_' + morph_op + str(kwargs.get('morph_kernel_size', 3))
    #                                              if morph_op != '' else ''))

    img_name = "{0}{1}{2}{3}{4}{5}{6}{7}".format(
        args[0].split('/')[-1].split('.')[0] if isinstance(args[0], str) else 'medianFilter',
        '_' + args[1],
        '_kernelSize' + str(kernel_size),
        '_sigma' + str(kwargs.get('sigma', -1))
        if kwargs.get('sigma', -1) != -1
        else '_computedSigma',
        '_threshold' + str(kwargs.get('threshold', 150)),
        '_sigmaRatio' + str(kwargs.get('dog_sigma_ratio', 1.6)),
        ('_colored' if kwargs.get('colorize', False) else ''),
        ('_' + morph_op + str(kwargs.get('morph_kernel_size', 3))
         if morph_op != '' else ''))

    # display image
    cv2.imshow(img_name, image)

    return image, img_name


def plot_images(images, titles, subtitle):
    num_images = len(images)

    # Create a figure object with enough subplots to hold all the images
    fig, axes = plt.subplots(nrows=1, ncols=num_images, figsize=(num_images * 4, 4))

    # Plot each image and set the title for each subplot
    for i in range(num_images):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].set_title(titles[i])
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    # Set the general subtitle for the entire figure
    fig.suptitle(subtitle, fontsize=14, fontweight='bold', y=1.05)

    # Adjust the spacing between the subplots and show the figure
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

    fig.savefig('../output_img/plotflowertiger.png')


def main():
    # All avaliable args: image path, DoG (Difference of gaussian) or LoG (Laplacian of Gaussian)
    # All available kwargs: kernel_size: int, sigma: float, DoG_sigma_ratio, threshold: int,
    # morphological_operation_type ['erode', 'dilate', 'opening', 'closing'], colorize: bool.
    # operation kernel size
    orig_img = cv2.imread('../Res/flower.jpg', cv2.IMREAD_COLOR)

    # mean filter
    median_filtered = median_filter(orig_img, kernel_size=7)
    cv2.imshow('median_filtered', median_filtered)

    image1, image1_name = action('../Res/flower.jpg', 'DoG', kernel_size=5, dog_sigma_ratio=1.6, threshold=150)
    image2, image2_name = action(median_filtered, 'DoG', kernel_size=5, dog_sigma_ratio=1.6, threshold=150)
    image3, image3_name = action('../Res/flower.jpg', 'LoG', kernel_size=5, threshold=150)
    image4, image4_name = action(median_filtered, 'LoG', kernel_size=5, threshold=150)
#
    imglist = [image1, image2, image3, image4]
    titles = ['DoG k_size=5, sigma_r=1.6', 'DoG k_size=5, sigma_r=1.6, Median filtered', 'LoG k_size=5',
              'LoG k_size=5, Median filtered']
    subtitle = 'sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8, kernel size = 9, threshold = 150'
    plot_images(imglist, titles, subtitle)
#
    # display images
    for i in range(1, 5):
        cv2.imshow(locals()['image' + str(i) + '_name'], locals()['image' + str(i)])

    #save images
    for i in range(1, 5):
      cv2.imwrite('../output_img/' + locals()['image' + str(i) + '_name'] + '.png', locals()['image' + str(i)])

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
