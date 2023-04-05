import numpy as np


def padding2(image, kernel, channels: int = 1, pad_type: str = 'constant') -> np.ndarray:
    """
    Pad the image with zeros.
    :param image: an image
    :param kernel: the kernel to convolute with the image
    :param pad_type: the type of padding to use
    :param channels: number of channels in the image
    :return: the padded image
    """
    # padding
    pad_height = kernel.shape[0] // 2
    pad_width = kernel.shape[1] // 2
    if channels == 1:
        padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), pad_type, constant_values=0)
    else:
        padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), pad_type,
                              constant_values=0)

    return padded_image


def pad(image, kernel, pad_type='constant'):
    # padding
    pad_height = kernel.shape[0] // 2
    pad_width = kernel.shape[1] // 2

    if len(image.shape) == 2:
        padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), pad_type, constant_values=0)
    else:
        padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), pad_type,
                              constant_values=0)

    return padded_image
