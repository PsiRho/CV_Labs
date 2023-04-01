import numpy as np


def colorize_edges(orig_img_color, segmented_img_grayscale):
    """
    Takes the colors from the original image and applies them to the segmented image where there are edges
    (white pixels). The func will create a boolean array where the white pixels are True and the rest are False.
    Then the array is used for selecting the pixels from the original image and apply them to the segmented image.

    :param orig_img_color: the original color image
    :param segmented_img_grayscale: the segmented image in grayscale / binary
    :return: the segmented image with the edges colored
    """
    white_pixels = segmented_img_grayscale == 255
    output = np.where(white_pixels[..., None], orig_img_color, 0)
    return output
