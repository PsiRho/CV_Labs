import cv2
import numpy as np


def gamma_correction(img, gamma):
    """
    This function applies gamma correction to an image.
    :param img: an image to apply gamma correction on
    :param gamma: gamma value. If gamma is less than 1, the image will be darker. If gamma is greater than 1, the image
    will be brighter. If gamma is 1, the image will be unchanged. A range of 0.5 to 2.5 is typically used.
    :return: the gamma corrected image
    """
    # Apply gamma correction using broadcasting
    img_gamma = (img / 255.0) ** (1 / gamma) * 255
    img_gamma = img_gamma.astype(np.uint8)

    return img_gamma


## Read the image
#test_Img = cv2.imread('../Res/tiger.jpg')
#
## Apply gamma correction
#gamma_corrected = gamma_correction(test_Img, 0.5)
#
## Show the images
#cv2.imshow('Original', test_Img)
#cv2.imshow('Gamma corrected', gamma_corrected)
#
## Wait for a key press
#cv2.waitKey(0)
#cv2.destroyAllWindows()
