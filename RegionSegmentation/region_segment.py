import cv2
import numpy as np
from Convolution.convolution import convolute
from Convolution.convolution import gaussian_kernel


def edge_detection_kernel(size: int = 3) -> np.ndarray:
    """
    Edge detection kernel based on the Laplacian of Gaussian.
    :param size: size of the kernel. e.g. 3, 5, 7, ...
    :return: the kernel to convolute with the image
    """
    # empty kernel
    kernel = np.zeros((size, size), dtype=np.float32)

    # set center to 1
    kernel[size // 2, size // 2] = 1

    # subtract gaussian kernel
    kernel = kernel - gaussian_kernel(size, 1, 1)
    print(kernel)  # TODO remove print

    return kernel


def region_segmentation(image: np.ndarray, kernel: np.ndarray, threshold: int = 0) -> np.ndarray:
    """
    Perform region segmentation on the given image.
    :param image: an image
    :param kernel: the kernel to convolute with the image
    :param threshold: the threshold to apply to the convoluted image
    :return: the segmented image
    """
    # convolute the image with the kernel
    convoluted_image = convolute(image, kernel, 1)

    # apply threshold
    segmented_image = np.where(convoluted_image > threshold, 255, 0)

    return segmented_image


def remove_isolated_pixels(image: np.ndarray, region_size: int) -> np.ndarray:
    """
    Remove isolated pixels from the image. A pixel is considered isolated if it has less than region_size white pixels
    in its 3x3 neighborhood.
    :param image: an image
    :param region_size: the minimum number of white pixels in the 3x3 neighborhood
    :return: the image with isolated pixels removed
    """
    # image dimensions
    i_row, i_col = image.shape[:2]

    # empty image for output
    output = np.zeros_like(image)

    # remove isolated pixels
    for y in range(i_row):
        for x in range(i_col):
            if image[y, x] == 255:
                # if pixel is not on the border
                if 0 < y < i_row - 1 and 0 < x < i_col - 1:
                    # if there are at least region_size white pixels in the 3x3 neighborhood
                    if (image[y - 1:y + 2, x - 1:x + 2] == 255).sum() >= region_size:
                        output[y, x] = 255

    return output


def colorize_edges(orig_img_color, segmented_img_grayscale):
    # get image dimensions
    i_row, i_col = orig_img_color.shape[:2]

    # create output image
    output = np.zeros_like(orig_img_color)

    # apply color to segmented image
    for y in range(i_row):
        for x in range(i_col):
            if segmented_img_grayscale[y, x] == 255:
                output[y, x] = orig_img_color[y, x]

    return output


# Wrapper for reading the image, performing region based segmentation and displaying the results.
def im_segment(image_path: str, kernel_size: int = 3, threshold: int = 0, region_size: int = 5,
               clean_image: bool = False):
    # read image
    orig_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # create edge detection kernel
    edge_detect_kern = edge_detection_kernel(kernel_size)

    # perform region based segmentation
    segmented_img_grayscale = region_segmentation(orig_img, edge_detect_kern, threshold)

    # remove unwanted regions
    if clean_image:
        segmented_img_grayscale = remove_isolated_pixels(segmented_img_grayscale, region_size)

    # convert images to uint8 for display
    segmented_img_grayscale = np.uint8(segmented_img_grayscale)

    # creates a "unique" name as imshow can't handle the same name twice
    img_name = image_path.split('/')[-1].split('.')[0]
    img_name = img_name + ' - ' + str(np.random.randint(0, 100))

    # display image
    cv2.imshow(img_name, segmented_img_grayscale)

    return segmented_img_grayscale


def main():
    # im_segment('../Res/flower.jpg', kernel_size=3, threshold=200, region_size=4)
    # im_segment('../Res/flower.jpg', kernel_size=3, threshold=200, region_size=4, clean_image=True)

    # read image
    orig_img_color = cv2.imread('../Res/flower.jpg', cv2.IMREAD_COLOR)
    orig_img_gray = cv2.cvtColor(orig_img_color, cv2.COLOR_BGR2GRAY)

    edge_detect_kern = edge_detection_kernel(3)
    seg_img_gray = region_segmentation(orig_img_gray, edge_detect_kern, 200)
    seg_img_cleaned = seg_img_gray.copy()
    seg_img_cleaned = remove_isolated_pixels(seg_img_cleaned, 4)

    colored = colorize_edges(orig_img_color, seg_img_gray)
    col_cleaned = colorize_edges(orig_img_color, seg_img_cleaned)

    cv2.imshow('Seg_colored', colored)
    cv2.imshow('Seg_colored_cleaned', col_cleaned)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()