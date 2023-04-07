import numpy as np
from Util.padding import pad


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
    padded_img = pad(image, kernel)

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
