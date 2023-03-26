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


