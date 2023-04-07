import numpy as np


# TODO: Refactor this function to use the histogram function from Histogram\histogram.py and use vectorization
#  instead of loops.
def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Perform histogram equalization on the given image.
    :param image: an image
    :return: the equalized image
    """
    # get image dimensions
    row, col = image.shape[:2]

    # create histogram
    histogram = np.zeros(256, dtype=np.float32)
    for y in range(row):
        for x in range(col):
            histogram[image[y, x]] += 1

    # normalize histogram
    histogram /= row * col

    # create cumulative histogram
    cumulative_histogram = np.zeros(256, dtype=np.float32)
    cumulative_histogram[0] = histogram[0]
    for i in range(1, 256):
        cumulative_histogram[i] = cumulative_histogram[i - 1] + histogram[i]

    # create lookup table
    lookup_table = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        lookup_table[i] = np.round(cumulative_histogram[i] * 255)

    # equalize image
    equalized_image = np.zeros_like(image)
    for y in range(row):
        for x in range(col):
            equalized_image[y, x] = lookup_table[image[y, x]]

    return equalized_image
