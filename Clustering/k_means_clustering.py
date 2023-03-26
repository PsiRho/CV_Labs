import numpy as np
import cv2


def initialize_clusters(img, k: int) -> np.ndarray:  # cluster_centers
    # choose random pixels as init cluster centers
    return img[np.random.choice(img.shape[0], k, replace=False), :]


def assign_labels(img, cluster_centers) -> np.ndarray:  # labels
    # assign each pixel to the closest cluster center
    distances = np.linalg.norm(img - cluster_centers[:, np.newaxis], axis=2)
    return np.argmin(distances, axis=0)

