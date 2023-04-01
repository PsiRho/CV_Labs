"""
This is a python script for learning image histograms without using any built-in functions.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
testImg = cv2.imread('../Res/flower.jpg', 0)


def histogram(img):
    """
    This function will create a histogram for the input image.
    :param img: The input image.
    :return: The histogram of the image.
    """
    # Create a histogram
    hist = np.zeros((256, 1), dtype=np.uint32)

    # Calculate the histogram
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            hist[img[i, j]] += 1

    return hist


# Plot the histogram
plt.plot(histogram(testImg))
plt.xlabel('Intensity')
plt.ylabel('Number of pixels')
plt.title('Histogram')
plt.show()
