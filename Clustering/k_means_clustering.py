import numpy as np
import cv2


def initialize_clusters(img, k: int) -> np.ndarray:  # cluster_centers
    # choose random pixels as init cluster centers
    return img[np.random.choice(img.shape[0], k, replace=False), :]