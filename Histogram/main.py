"""
This is a python script for learning image histograms without using any built-in functions.
"""

import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

# Read the image
img = cv2.imread('flower.jpg', 0)

# Create a histogram
hist = np.zeros((256, 1), dtype=np.int32)

# Calculate the histogram
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        hist[img[i, j]] += 1

# Plot the histogram
plt.plot(hist)
plt.xlabel('Intensity')
plt.ylabel('Number of pixels')
plt.title('Histogram')
plt.show()
