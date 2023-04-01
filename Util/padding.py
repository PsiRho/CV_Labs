import numpy as np


def padding(image, kernel) -> np.ndarray:
    """
    Pad the image with zeros.
    :param image: an image
    :param kernel: the kernel to convolute with the image
    :return: the padded image
    """
    # padding
    pad_height = kernel.shape[0] // 2
    pad_width = kernel.shape[1] // 2
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), 'constant')

    return padded_image
