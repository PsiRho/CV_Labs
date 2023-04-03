import time
import timeit

import cv2
import numpy as np
from matplotlib import pyplot as plt
import Convolution.convolution as conv
import MorphologicalOps.morphop as mrph
import Util.colorize as clr


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
    convoluted_image = conv.convolute(image, kernel, channels=1)

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
    convoluted_image1 = conv.convolute(image, kernel1, 1)  # Image with low SD
    convoluted_image2 = conv.convolute(image, kernel2, 1)  # Image with high SD
    diff_of_gaussian = convoluted_image2 - convoluted_image1

    # threshold
    diff_of_gaussian = np.where(diff_of_gaussian >= threshold, 255, 0)

    return diff_of_gaussian


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
    # read image
    orig_img = cv2.imread(args[0], cv2.IMREAD_COLOR)

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

    # get image name, TODO: make a separate function for this
    img_name = "{0}{1}{2}{3}{4}{5}{6}{7}".format(args[0].split('/')[-1].split('.')[0],
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

    fig.savefig('../Res/plotflowertiger.png')


def main():
    # All avaliable args: image path, DoG (Difference of gaussian) or LoG (Laplacian of Gaussian)
    # All available kwargs: kernel_size: int, sigma: float, DoG_sigma_ratio, threshold: int,
    # morphological_operation_type ['erode', 'dilate', 'opening', 'closing'], colorize: bool.
    # operation kernel size
    image1, image1_name = action('../Res/flower.jpg', 'DoG', kernel_size=5, dog_sigma_ratio=1.6, threshold=150)
    image2, image2_name = action('../Res/tiger.jpg', 'DoG', kernel_size=5, dog_sigma_ratio=1.6, threshold=150)
    image3, image3_name = action('../Res/flower.jpg', 'LoG', kernel_size=5, threshold=150)
    image4, image4_name = action('../Res/tiger.jpg', 'LoG', kernel_size=5, threshold=150)

    imglist = [image1, image3, image2, image4]
    titles = ['DoG k_size=5, sigma_r=1.6', 'LoG k_size=5', 'DoG k_size=5, sigma_r=1.6',
              'LoG k_size=5']
    subtitle = 'sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8, kernel size = 9, threshold = 150'
    plot_images(imglist, titles, subtitle)

    # save images
    #for i in range(1, 5):
    #   cv2.imwrite('../Res/' + locals()['image' + str(i) + '_name'] + '.png', locals()['image' + str(i)])

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
