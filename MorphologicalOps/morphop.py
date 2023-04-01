import numpy as np
from Util.padding import padding


def morph_op(image, op_type: str, kernel_size: int = 3) -> np.ndarray:
    """
    Perform morphological operation on the given image.
    :param image: an image
    :param op_type: type of the operation. Either 'erode' or 'dilate'
    :param kernel_size: size of the kernel
    :return: the segmented image
    """
    # erosion kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # padding
    padded_img = padding(image, kernel)

    # output image
    output = np.zeros_like(image)

    # erode image
    for y in range(output.shape[0]):
        for x in range(output.shape[1]):
            patch = padded_img[y:y + kernel.shape[0], x:x + kernel.shape[1]]
            if op_type == 'erode' and np.all(patch * kernel == 255):
                output[y, x] = 255
            if op_type == 'dilate' and np.any(patch * kernel == 255):
                output[y, x] = 255

    return output


def opening(image, kernel_size: int = 3) -> np.ndarray:
    """
    Perform opening on the given image.
    :param kernel_size: size of the kernel
    :param image: an image
    :return: the segmented image
    """
    # erode image
    eroded_image = morph_op(image, 'erode', kernel_size)

    # dilate image
    opened_img = morph_op(eroded_image, 'dilate', kernel_size)

    return opened_img


def closing(image, kernel_size: int = 3) -> np.ndarray:
    """
    Perform closing on the given image.
    :param kernel_size: size of the kernel
    :param image: an image
    :return: the segmented image
    """
    # dilate image
    dilated_image = morph_op(image, 'dilate', kernel_size)

    # erode image
    closed_img = morph_op(dilated_image, 'erode', kernel_size)

    return closed_img
