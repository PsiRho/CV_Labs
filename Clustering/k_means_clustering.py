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


