"""
This is a python script for learning gamma correction in python. It will use the histogram function from the histogram
folder.
"""

import cv2
import numpy as np

# Read the image
testImg = cv2.imread('../Res/tiger.jpg', 0)


# Create a gamma correction function
def gamma_correction(img, gamma):
    """
    This function will apply gamma correction to the input image.
    :param img: The input image.
    :param gamma: The gamma value.
    :return: The gamma corrected image.
    """
    # Create a lookup table
    lookup_table = np.zeros((256, 1), dtype=np.uint8)

    # Fill the lookup table
    for i in range(256):
        lookup_table[i] = 255 * pow(i / 255, 1 / gamma)

    # Apply the gamma correction
    img_gamma = cv2.LUT(img, lookup_table)

    return img_gamma


# Apply gamma correction
gamma_corrected = gamma_correction(testImg, 0.75)

# Show the images
cv2.imshow('Original', testImg)
cv2.imshow('Gamma corrected', gamma_corrected)

# Wait for a key press
cv2.waitKey(0)
cv2.destroyAllWindows()
