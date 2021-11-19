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

def normalize(value, rmin, rmax, tmin, tmax):
    rmin = float(rmin)
    rmax = float(rmax)
    tmin = float(tmin)
    tmax = float(tmax)
    return ((value - rmin) / (rmax - rmin)) * (tmax - tmin) + tmin

def frechet_distance(traj1, traj2, config):
    p = deepcopy(np.asarray(traj1))
    q = deepcopy(np.asarray(traj2))
    p[:, 0] = normalize(p[:, 0],
                       config['DEFAULT']['rmin_x'],
                       config['DEFAULT']['rmax_x'],
                       config['DEFAULT']['tmin_x'],
                       config['DEFAULT']['tmax_x'])
    p[:, 1] = normalize(p[:, 1],
                        config['DEFAULT']['rmin_y'],
                        config['DEFAULT']['rmax_y'],
                        config['DEFAULT']['tmin_y'],
                        config['DEFAULT']['tmax_y'])
    p[:, 2] = normalize(p[:, 2],
                        config['DEFAULT']['rmin_z'],
                        config['DEFAULT']['rmax_z'],
                        config['DEFAULT']['tmin_z'],
                        config['DEFAULT']['tmax_z'])
    p = reduce_traj(0, p, config)
    q[:, 0] = normalize(q[:, 0],
                       config['DEFAULT']['rmin_x'],
                       config['DEFAULT']['rmax_x'],
                       config['DEFAULT']['tmin_x'],
                       config['DEFAULT']['tmax_x'])
    q[:, 1] = normalize(q[:, 1],
                        config['DEFAULT']['rmin_y'],
                        config['DEFAULT']['rmax_y'],
                        config['DEFAULT']['tmin_y'],
                        config['DEFAULT']['tmax_y'])
    q[:, 2] = normalize(q[:, 2],
                        config['DEFAULT']['rmin_z'],
                        config['DEFAULT']['rmax_z'],
                        config['DEFAULT']['tmin_z'],
                        config['DEFAULT']['tmax_z'])
    q = reduce_traj(0, q, config)
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

def reduce_traj(index, trajectory, config):
    traj = np.asarray(trajectory)[:, :3]

    traj[:, 0] = normalize(traj[:, 0],
                        config['DEFAULT']['rmin_x'],
                        config['DEFAULT']['rmax_x'],
                        config['DEFAULT']['tmin_x'],
                        config['DEFAULT']['tmax_x'])
    traj[:, 1] = normalize(traj[:, 1],
                        config['DEFAULT']['rmin_y'],
                        config['DEFAULT']['rmax_y'],
                        config['DEFAULT']['tmin_y'],
                        config['DEFAULT']['tmax_y'])
    traj[:, 2] = normalize(traj[:, 2],
                        config['DEFAULT']['rmin_z'],
                        config['DEFAULT']['rmax_z'],
                        config['DEFAULT']['tmin_z'],
                        config['DEFAULT']['tmax_z'])
    new_traj, indices = rdp_with_index(traj, range(np.shape(traj)[0]), 50)
    new_traj = np.asarray(new_traj)
    return new_traj

def cluster_trajectories(trajs, latents, means, config, num_clusters=20):

    trajectories = deepcopy(trajs)
    np.asarray(trajectories)
    num_cores = multiprocessing.cpu_count()
    all_reduced_trajectories = Parallel(n_jobs=num_cores)(
        delayed(reduce_traj)(i, traj, config)
       for i, traj in enumerate(trajectories))

    dist_matrix = np.zeros((len(all_reduced_trajectories), len(all_reduced_trajectories)))
    dist_matrices = Parallel(n_jobs=num_cores)(
        delayed(thread_compute_distance)(i, traj, all_reduced_trajectories)
       for i, traj in enumerate(all_reduced_trajectories))

    for dist in dist_matrices:
        for key in dist.keys():
            indeces = [int(k) for k in key.split(',')]
            dist_matrix[indeces[0], indeces[1]] += dist[key]

    dist_matrix = np.clip(dist_matrix, 0, 9999999)
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