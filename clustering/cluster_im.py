
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.metrics import pairwise_distances_argmin_min
import hdbscan
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

# Cluster through latent space of autoencoder.
# The autoencoder is made in Torch
# TODO: move the autoencoder to TF
def cluster(im_rews, clustering_mode='kmeans', clusters=10, means=None, sums=None):

    # The Imitation rewards will be our latent space to cluster
    latents = np.asarray(im_rews)

    # Cluster through Spectral
    if clustering_mode == 'spectral':
        clusterer = SpectralClustering(clusters).fit(latents)
        closest = []
        num_cluster = np.max(clusterer.labels_)
        for i in range(num_cluster + 1):
            index = np.where(clusterer.labels_ == i)[0][0]
            closest.append(index)

    elif clustering_mode == 'kmeans':
        clusterer = KMeans(clusters).fit(latents)
        num_cluster = np.max(clusterer.labels_)
        closest, _ = pairwise_distances_argmin_min(clusterer.cluster_centers_, latents)
        #
    elif clustering_mode == 'hdbscan':
        clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=1).fit(latents)
        closest = []
        num_cluster = np.max(clusterer.labels_)
        for i in range(num_cluster + 1):
            index = np.where(clusterer.labels_ == i)[0][0]
            closest.append(index)
    elif clustering_mode == 'hac':
        clusterer = AgglomerativeClustering(clusters).fit(latents)
        num_cluster = np.max(clusterer.labels_)
        # closest, _ = pairwise_distances_argmin_min(clusterer.cluster_centers_, latents)

    closest = []
    for l in range(num_cluster + 1):
        labels_indexes = np.where(clusterer.labels_ == l)[0]
        cluster_latents = latents[labels_indexes]
        cluster_means = means[labels_indexes]
        # highest_peak_traj_index_in_cluster = np.unravel_index(np.argmax(cluster_latents), np.shape(cluster_latents))[0]
        # highest_peak_traj_index_in_cluster = np.argmax(np.sum(cluster_latents, axis=1))
        # peaks = np.max(cluster_latents, axis=1)
        cluster_means = cluster_means

        highest_peak_traj_index_in_cluster = np.argmax(cluster_means)
        highest_peak_latent = np.reshape(cluster_latents[highest_peak_traj_index_in_cluster], (1, -1))
        highest_peak_traj_index, _ = pairwise_distances_argmin_min(highest_peak_latent, latents)
        closest.append(highest_peak_traj_index[0])

    closest = np.asarray(closest)
    # Return the indices of the trajectories that define each cluster
    print('Clustering done! Num cluster: {}'.format(num_cluster + 1))
    return np.asarray(closest), None, clusterer.labels_

if __name__ == '__main__':

    trajectories = np.load('../traj_to_observe.npy')
    cluster(trajectories, 'autoencoders/labyrinth')