import cv2


def thresholding(img, threshold):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j][0] <= threshold:
                img[i][j] = 0
    return img


test_img = cv2.imread('../Res/flower.jpg', cv2.IMREAD_COLOR)
thresh_img = thresholding(test_img, 40)


cv2.imshow('thresh_img', thresh_img)

# Wait for a key press
cv2.waitKey(0)
cv2.destroyAllWindows()
