import cv2

test_img = cv2.imread('Res/flower.jpg', cv2.IMREAD_COLOR)


def thresholding(img, low_threshold, high_threshold):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j][0] <= low_threshold:
                img[i][j] = 16
            elif img[i][j][0] >= high_threshold:
                img[i][j] = 200
    return img


thresh_img = thresholding(test_img, 79, 140)

cv2.imshow('thresh_img', thresh_img)

# Wait for a key press
cv2.waitKey(0)
cv2.destroyAllWindows()
