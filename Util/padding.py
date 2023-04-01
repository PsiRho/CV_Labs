import numpy as np


def padding(image, kernel, channels: int = 1) -> np.ndarray:
    """
    Pad the image with zeros.
    :param image: an image
    :param kernel: the kernel to convolute with the image
    :param channels: number of channels in the image
    :return: the padded image
    """
    # padding
    pad_height = kernel.shape[0] // 2
    pad_width = kernel.shape[1] // 2
    if channels == 1:
        padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), 'constant', constant_values=0)
    else:
        padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), 'constant',
                              constant_values=0)

    return padded_image
