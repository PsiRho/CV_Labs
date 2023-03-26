import numpy as np
import cv2


def initialize_clusters(img, k: int) -> np.ndarray:  # cluster_centers
    # choose random pixels as init cluster centers
    return img[np.random.choice(img.shape[0], k, replace=False), :]


def assign_labels(img, cluster_centers) -> np.ndarray:  # labels
    # assign each pixel to the closest cluster center
    distances = np.linalg.norm(img - cluster_centers[:, np.newaxis], axis=2)
    return np.argmin(distances, axis=0)


def update_clusters(img, labels, k: int) -> np.ndarray:
    # update centers based on the mean of the pixels assigned to each cluster
    new_cluster_centers = np.zeros((k, 3))
    for i in range(k):
        if np.sum(labels == i) == 0:  # if cluster is empty
            new_cluster_centers[i] = img[np.random.choice(img.shape[0], 1), :]  # choose random pixel as new center
        else:
            new_cluster_centers[i] = np.mean(img[labels == i], axis=0)  # update center
    return new_cluster_centers


def k_means_clustering(image, k: int, tolerance: float) -> (np.ndarray, np.ndarray):
    # perform k-means clustering on the image
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # convert to HSV
    height, width = hsv_image.shape[:2]
    img = hsv_image.reshape((height * width, 3))
    cluster_centers = initialize_clusters(img, k)
    iterations = 0
    while True:
        labels = assign_labels(img, cluster_centers)
        new_cluster_centers = update_clusters(img, labels, k)
        iterations += 1
        if np.allclose(cluster_centers, new_cluster_centers, rtol=tolerance):  # check for convergence
            break
        cluster_centers = new_cluster_centers
    print(f'k-means clustering converged after {iterations} iterations.')
    labels_2d = labels.reshape((height, width))
    return labels_2d, cluster_centers


def create_image_from_labels(labels, cluster_centers) -> np.ndarray:
    # puts together an image from the labels and cluster centers
    height, width = labels.shape[:2]
    image = np.zeros((height, width, 3))
    for i in range(height):
        for j in range(width):
            image[i, j] = cluster_centers[labels[i, j]]

    image = image.astype(np.uint8)  # convert to uint8
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


# wrapper for timing functions with args
def timeit_wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)

    return wrapped


def main():
    # read image
    orig_img = cv2.imread('../Res/flower.jpg', cv2.IMREAD_COLOR)

    # perform k-means clustering on the image
    labels, cluster_centers = k_means_clustering(orig_img, k=10, tolerance=1e-4)
    #
    # create image from labels and cluster centers
    clustered_img = create_image_from_labels(labels, cluster_centers)

    # show images
    cv2.imshow('Original Image', orig_img)
    cv2.imshow('Clustered Image', clustered_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
