import timeit

import matplotlib
import numpy as np
import cv2

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.spatial.distance import cdist

matplotlib.use('TkAgg')


def random_pixel(img, k: int) -> np.ndarray:  # cluster_centers
    flat_img = img.reshape(-1, img.shape[-1])
    return img[np.random.choice(flat_img.shape[0], k, replace=False), :]


def initialize_clusters(img, k: int) -> np.ndarray:
    # random pixels as init center
    cluster_centers = np.zeros((k, 3))
    cluster_centers[0] = random_pixel(img, 1)
    for i in range(1, k):
        distances = np.linalg.norm(img - cluster_centers[i - 1, np.newaxis], axis=1)  # distance from previous center
        cluster_centers[i] = img[np.argmax(distances), :]  # pixel farthest from previous center
    return cluster_centers


def assign_labels2(img, cluster_centers) -> np.ndarray:
    # assign each pixel to the closest cluster center
    distances = np.linalg.norm(img - cluster_centers[:, np.newaxis], axis=2)
    return np.argmin(distances, axis=0)


def assign_labels(img, cluster_centers) -> np.ndarray:
    # assign each pixel to the closest cluster center
    distances = cdist(img, cluster_centers)
    return np.argmin(distances, axis=1)


def update_clusters2(img, labels, k: int) -> np.ndarray:
    # update centers based on the mean of the pixels assigned to each cluster
    new_cluster_centers = np.zeros((k, 3))
    for i in range(k):
        if np.sum(labels == i) == 0:  # if cluster is empty
            new_cluster_centers[i] = random_pixel(img, 1)  # new random center
        else:
            new_cluster_centers[i] = np.mean(img[labels == i], axis=0)  # update center
    return new_cluster_centers


def update_clusters(img, labels, k: int) -> np.ndarray:
    # update centers based on the mean of the pixels assigned to each cluster
    new_cluster_centers = np.zeros((k, 3))
    for i in range(k):
        mask = labels == i
        count = np.sum(mask)
        if count == 0:  # if cluster is empty
            new_cluster_centers[i] = random_pixel(img, 1)  # new random center
        else:
            new_cluster_centers[i] = np.sum(img * mask[:, np.newaxis], axis=0) / count
    return new_cluster_centers


def k_means_clustering(image, k: int, tolerance: float) -> (np.ndarray, np.ndarray):
    # perform k-means clustering on the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    height, width = image.shape[:2]
    img = image.reshape((height * width, 3))
    cluster_centers = initialize_clusters(img, k)
    iterations = 0
    while True:
        iterations += 1
        labels = assign_labels(img, cluster_centers)
        new_cluster_centers = update_clusters(img, labels, k)
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

    image = image.astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


# wrapper for timing functions with args
def timeit_wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)

    return wrapped


def scatterplot(image, labels, cluster_centers):
    # Reshape the image data to a 2D array
    reshaped_image = np.reshape(image, (-1, 3))

    # create a colormap with the colors of the cluster centers
    colormap = ListedColormap(cluster_centers / 255)

    # Create scatter plot of the k-means results
    plt.scatter(reshaped_image[:, 0], reshaped_image[:, 1], c=labels, cmap=colormap)
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='black', marker='x', s=100)
    plt.title(f'K-means clustering with {len(cluster_centers)} clusters')
    plt.show()


def display_kmeans_clusters(img, labels, cluster_centers):
    num_clusters = len(cluster_centers)
    fig, axes = plt.subplots(nrows=(num_clusters + 3) // 4, ncols=min(num_clusters, 4), figsize=(10, 10))
    axes = axes.flatten()

    # Computes the number of pixels in each cluster
    cluster_sizes = np.bincount(labels.flatten())
    sorted_indices = np.argsort(cluster_sizes)[::-1]
    cluster_centers = cluster_centers[sorted_indices]
    # cluster_sizes as a percentage of total pixels
    cluster_sizes = cluster_sizes[sorted_indices] / np.prod(img.shape[:2]) * 100

    for i, center in enumerate(cluster_centers):
        mask = labels == i
        masked_img = np.zeros_like(img, dtype=np.int32)
        masked_img += mask[..., np.newaxis] * center.astype(np.int32)
        masked_img = np.clip(masked_img, 0, 255).astype(np.uint8)
        axes[i].imshow(masked_img)
        axes[i].set_xticks([])
        axes[i].set_yticks([])

        # gets the RGB values of the cluster center
        center = np.round(center).astype(np.uint8)

        # Set title to Cluster #, percentage of pixels, and RGB values
        axes[i].set_title(
            f'Pixels in cluster: ({cluster_sizes[i]:.2f}%)\n' + f'[R: {center[0]}, G: {center[1]}, B: {center[2]}]')

    # Hide remaining axes
    for ax in axes[num_clusters:]:
        ax.remove()

    plt.tight_layout()
    plt.show()


def main():
    # read image
    orig_img = cv2.imread('../Res/tiger.jpg', cv2.IMREAD_COLOR)

    # perform k-means clustering on the image
    labels, cluster_centers = k_means_clustering(orig_img, k=8, tolerance=1e-4)

    clustered_img = create_image_from_labels(labels, cluster_centers)

    cv2.imshow('clustered image', clustered_img)

    # for i in range(5):
    #    labels, cluster_centers = k_means_clustering(orig_img, k=10, tolerance=1e-2)
    #    clustered_img = create_image_from_labels(labels, cluster_centers)
    #    cv2.imshow(f'clustered image {i}', clustered_img)

    # display_kmeans_clusters(orig_img, labels, cluster_centers)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
