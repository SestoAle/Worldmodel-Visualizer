import numpy as np
from clustering.rdp import rdp_with_index, distance
from joblib import Parallel, delayed
import multiprocessing
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn_extra.cluster import KMedoids
from clustering.distances.discrete import FastDiscreteFrechetMatrix, FastDiscreteFrechetSparse, \
    earth_haversine, euclidean
from copy import deepcopy
import time

def compute_distance_matrix(trajectories, method="Frechet"):
    """
    :param method: "Frechet" or "Area"
    """
    n = len(trajectories)
    dist_m = np.zeros((n, n))
    distance = euclidean
    fdfdm = FastDiscreteFrechetSparse(distance)
    for i in range(n - 1):
        p = trajectories[i]
        for j in range(i + 1, n):
            q = trajectories[j]
            if method == "Frechet":
                dist_m[i, j] = fdfdm.distance(p, q)
            else:
                dist_m[i, j] = similaritymeasures.area_between_two_curves(p, q)
            dist_m[j, i] = dist_m[i, j]
    return dist_m

def frechet_distance(traj1, traj2):
    p = deepcopy(traj1)
    q = deepcopy(traj2)
    p[:, 0] = (((p[:, 0]) + 1) / 2) * 500
    p[:, 1] = (((p[:, 1]) + 1) / 2) * 500
    p[:, 2] = (((p[:, 2]) + 1) / 2) * 60
    p = reduce_traj(0, p)
    q[:, 0] = (((q[:, 0]) + 1) / 2) * 500
    q[:, 1] = (((q[:, 1]) + 1) / 2) * 500
    q[:, 2] = (((q[:, 2]) + 1) / 2) * 60
    q = reduce_traj(0, q)
    distance = euclidean
    fdfdm = FastDiscreteFrechetMatrix(distance)
    return fdfdm.distance(p, q)

def thread_compute_distance(index, trajectory, trajectories):
    n = len(trajectories)
    p = trajectory
    distances = dict()
    distance = euclidean
    fdfdm = FastDiscreteFrechetMatrix(distance)
    for j in range(index + 1, n):
        q = trajectories[j]
        distances['{},{}'.format(index,j)] = distances['{},{}'.format(j,index)] = fdfdm.distance(p, q)
    return distances

def reduce_traj(index, trajectory):
    traj = trajectory[:, :3]
    traj[:, 0] = (((traj[:, 0]) + 1) / 2) * 500
    traj[:, 1] = (((traj[:, 1]) + 1) / 2) * 500
    traj[:, 2] = (((traj[:, 2]) + 1) / 2) * 60
    new_traj, indices = rdp_with_index(traj, range(np.shape(traj)[0]), 50)
    new_traj = np.asarray(new_traj)
    return new_traj

def cluster_trajectories(trajs, latents, means, num_clusters=20):

    trajectories = deepcopy(trajs)
    np.asarray(trajectories)
    num_cores = multiprocessing.cpu_count()
    all_reduced_trajectories = Parallel(n_jobs=num_cores)(
        delayed(reduce_traj)(i, traj)
       for i, traj in enumerate(trajectories))

    dist_matrix = np.zeros((len(all_reduced_trajectories), len(all_reduced_trajectories)))
    dist_matrices = Parallel(n_jobs=num_cores)(
        delayed(thread_compute_distance)(i, traj, all_reduced_trajectories)
       for i, traj in enumerate(all_reduced_trajectories))

    for dist in dist_matrices:
        for key in dist.keys():
            indeces = [int(k) for k in key.split(',')]
            dist_matrix[indeces[0], indeces[1]] += dist[key]

    clusterer = KMedoids(num_clusters, metric='precomputed', init='k-medoids++').fit(dist_matrix)

    num_cluster = np.max(clusterer.labels_)
    closest = []
    latents = np.asarray(latents)
    for l in range(num_cluster + 1):
        labels_indexes = np.where(clusterer.labels_ == l)[0]
        cluster_latents = latents[labels_indexes]
        cluster_means = means[labels_indexes]
        cluster_means = cluster_means

        highest_peak_traj_index_in_cluster = np.argmax(cluster_means)
        highest_peak_latent = np.reshape(cluster_latents[highest_peak_traj_index_in_cluster], (1, -1))
        highest_peak_traj_index, _ = pairwise_distances_argmin_min(highest_peak_latent, latents)
        closest.append(highest_peak_traj_index[0])

    closest = np.asarray(closest)

    return closest, None, clusterer.labels_