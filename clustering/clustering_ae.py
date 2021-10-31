import torch
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import pairwise_distances_argmin_min
import hdbscan
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

# Cluster through latent space of autoencoder.
# The autoencoder is made in Torch
# TODO: move the autoencoder to TF
def cluster(trajectories, model, clustering_mode='kmeans', clusters=35):

    print('Clustering trajectories...')
    # Load pre-trained autoencoder
    autoencoder = torch.load(model)
    print('AutoEncoders loaded!')

    # Get the trajectories
    trajectories = trajectories[:, :, :3]
    # trajectories = preprocessing(real_trajectories, 10, 5)

    # Pass the trajectories in the autoencoder to get the latent vectors
    # Divide the trajectories in mini-batches
    size = np.shape(trajectories)
    num_batch = 10
    mini_batch_len = int(size[0] / num_batch)
    latents = None
    for i in range(num_batch):
        trajectories_tensor = trajectories[i * mini_batch_len: i * mini_batch_len + mini_batch_len, :, :]
        trajectories_tensor = torch.from_numpy(trajectories_tensor)
        trajectories_tensor = trajectories_tensor.cuda()
        trajectories_tensor = trajectories_tensor.float()
        _, tmp_latents = autoencoder.forward(trajectories_tensor)
        tmp_latents = tmp_latents[0].cpu().detach().numpy()
        tmp_latents = np.reshape(tmp_latents, [-1, 512])
        if latents is None:
            latents = tmp_latents
        else:
            latents = np.concatenate([latents, tmp_latents], axis=0)

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
        closest = np.asarray(closest)
    elif clustering_mode == 'hdbscan':
        clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=1).fit(latents)
        closest = []
        num_cluster = np.max(clusterer.labels_)
        for i in range(num_cluster + 1):
            index = np.where(clusterer.labels_ == i)[0][0]
            closest.append(index)

    # Return the indices of the trajectories that define each cluster
    print('Clustering done! Num cluster: {}'.format(num_cluster))
    return np.asarray(closest)

if __name__ == '__main__':

    trajectories = np.load('../traj_to_observe.npy')
    cluster(trajectories, 'autoencoders/labyrinth')