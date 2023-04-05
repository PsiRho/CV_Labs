import numpy as np


# TODO: Refactor this function to use the histogram module from the Histogram package
def contrast_stretching(image: np.ndarray) -> np.ndarray:
    """
    Perform contrast stretching on the given image.
    :param image: an image to perform contrast stretching on
    :return: the stretched image
    """
    # Create histogram
    histogram, _ = np.histogram(image.ravel(), bins=256, range=(0, 255))

    # Normalize histogram
    histogram = histogram / np.sum(histogram)

    # Create cumulative histogram
    cumulative_histogram = np.cumsum(histogram)

    # Create lookup table
    lookup_table = np.round(255 * (cumulative_histogram - cumulative_histogram.min()) / (
                cumulative_histogram.max() - cumulative_histogram.min()))

    # Stretch image using lookup table
    stretched_image = lookup_table[image]

    return stretched_image.astype(np.uint8)
