import math
import timeit

import matplotlib
import numpy as np
import cv2

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.spatial.distance import cdist

matplotlib.use('Qt5Agg')


# ##################################################K-MEANS CLUSTERING##################################################


def random_pixel(img, k: int) -> np.ndarray:
    flat_img = img.reshape(-1, img.shape[-1])
    return img[np.random.choice(flat_img.shape[0], k, replace=False), :]


def initialize_clusters(img, k: int) -> np.ndarray:
    cluster_centers = np.zeros((k, 3))
    cluster_centers[0] = random_pixel(img, 1)  # random center
    for i in range(1, k):
        distances = cdist(img, np.array([cluster_centers[i - 1]]))[:, 0]  # distance from previous center
        cluster_centers[i] = img[np.argmax(distances), :]  # pixel farthest from previous center
    return cluster_centers


def update_clusters(img, labels, k: int) -> np.ndarray:
    # update centers based on the mean of the pixels assigned to each cluster
    new_cluster_centers = np.zeros((k, 3))
    for i in range(k):
        if np.sum(labels == i) != 0:
            new_cluster_centers[i] = np.mean(img[labels == i], axis=0)  # update center to mean of pixels
        else:  # if cluster is empty
            new_cluster_centers[i] = random_pixel(img, 1)  # new random center
    return new_cluster_centers


def assign_labels(img, cluster_centers) -> np.ndarray:
    # assign each pixel to the closest cluster center
    distances = cdist(img, cluster_centers)
    return np.argmin(distances, axis=1)


def k_means_clustering(image, k: int, tolerance: float) -> (np.ndarray, np.ndarray):
    # perform k-means clustering on the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
    # prints the number of iterations it took to converge and the number of clusters
    print(f'k-means clustering converged after {iterations} iterations with {k} clusters.')
    labels_2d = labels.reshape((height, width))

    return labels_2d, cluster_centers


# ###################################################UTILITY FUNCTIONS##################################################


def create_result_img(labels, cluster_centers) -> np.ndarray:
    # puts together an image from the labels and cluster centers
    height, width = labels.shape[:2]
    image = cluster_centers[labels.flatten()].reshape((height, width, -1))
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
    fig, axes = plt.subplots(nrows=(num_clusters + 3) // 4 + 1, ncols=min(num_clusters + 1, 4), figsize=(10, 10))
    axes = axes.flatten()

    # Add the original image as the first image
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    axes[0].imshow(img)
    axes[0].set_title("Clustered Image")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Computes the number of pixels in each cluster
    cluster_sizes = np.bincount(labels.flatten())

    # cluster_sizes as a percentage of total pixels
    cluster_sizes = cluster_sizes / np.prod(img.shape[:2]) * 100

    for i, center in enumerate(cluster_centers):
        mask = labels == i
        masked_img = np.zeros_like(img, dtype=np.int32)
        masked_img += mask[..., np.newaxis] * center.astype(np.int32)
        masked_img = np.clip(masked_img, 0, 255).astype(np.uint8)
        axes[i+1].imshow(masked_img)
        axes[i+1].set_xticks([])
        axes[i+1].set_yticks([])

        # gets the RGB values of the cluster center
        center = np.round(center).astype(np.uint8)

        # Set title to Cluster #, percentage of pixels, and RGB values
        axes[i+1].set_title(
            f'Pixels in cluster: ({cluster_sizes[i]:.2f}%)\n' + f'[R: {center[0]}, G: {center[1]}, B: {center[2]}]')

    # Hide remaining axes
    for ax in axes[num_clusters+1:]:
        ax.remove()

    plt.tight_layout()
    #plt.savefig('../output_img/kmeans/tiger_plotkmeans_separateclusters.jpg')
    plt.show()


def display_kmeans_results(original_image, result_images, k_values):
    num_clusters = len(k_values)
    num_images = num_clusters + 1

    # Calculate the number of rows and columns for the subplot arrangement
    num_cols = int(math.sqrt(num_images-1))+1
    num_rows = int(math.ceil(num_images/num_cols))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 5*num_rows))

    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    for i in range(num_clusters):
        result_images[i] = cv2.cvtColor(result_images[i], cv2.COLOR_BGR2RGB)

    # Display the cluster images in the subplots
    for i in range(num_clusters):
        r, c = divmod(i, num_cols)
        axes[r, c].set_title(f"k = {k_values[i]}")
        axes[r, c].imshow(result_images[i], cmap='inferno')
        axes[r, c].axis('off')

    # Display the original image in the bottom-right subplot
    r, c = divmod(num_images-1, num_cols)
    axes[r, c].set_title("Original image")
    axes[r, c].imshow(original_image, cmap='inferno')
    axes[r, c].axis('off')

    # Remove any empty subplots
    for i in range(num_images, num_rows*num_cols):
        r, c = divmod(i, num_cols)
        fig.delaxes(axes[r, c])

    plt.tight_layout()
    #plt.savefig('../output_img/kmeans/flower_plotkmeans.jpg')
    plt.show()




def main():
    # read image
    orig_img = cv2.imread('../Res/tiger.jpg', cv2.IMREAD_COLOR)

    # perform k-means clustering on the image
    labels, cluster_centers = k_means_clustering(orig_img, k=7, tolerance=1e-4)

    clustered_img = create_result_img(labels, cluster_centers)

    #cv2.imwrite('../output_img/kmeans/tiger_kmeans_7_forplot.jpg', clustered_img)

    #cv2.imshow('clustered image', clustered_img)

    #images = []
    #labels = None
    #cluster_centers = None
    # Will loop from 2 to 32 with a step of 2**i
    #for i in range(1, 6):
    #    labels, cluster_centers = k_means_clustering(orig_img, k=2**i, tolerance=1e-4)
    #    clustered = create_result_img(labels, cluster_centers)
    #    images.append(clustered)
    #    cv2.imwrite(f'../output_img/kmeans/flower_kmeans_{2**i}.jpg', clustered)

    #display_kmeans_results(orig_img, images, [2**i for i in range(1, 6)])

    display_kmeans_clusters(clustered_img, labels, cluster_centers)

    # create images with each cluster as a separate image and save them
    #for i in range(len(cluster_centers)):
    #    mask = labels == i
    #    masked_img = np.zeros_like(orig_img, dtype=np.int32)
    #    masked_img += mask[..., np.newaxis] * cluster_centers[i].astype(np.int32)
    #    masked_img = np.clip(masked_img, 0, 255).astype(np.uint8)
    #    cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB, masked_img)
    #    cv2.imwrite(f'../output_img/kmeans/tiger_kmeans_{i}_separatedClusters.jpg', masked_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
