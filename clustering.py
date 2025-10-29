import os
import numpy as np
from sklearn.cluster import DBSCAN


def dbscan_on_voxels(x_centers, y_centers, counts, eps=0.6, min_samples=3, metric='euclidean'):
    """Applique DBSCAN sur les centres de voxels dont counts>0. Retourne labels, mask, summary."""
    x_centers = np.asarray(x_centers)
    y_centers = np.asarray(y_centers)
    counts = np.asarray(counts)
    Xg, Yg = np.meshgrid(x_centers, y_centers)
    mask = counts > 0
    pts = np.column_stack((Xg[mask], Yg[mask]))
    if pts.shape[0] == 0:
        raise ValueError('Aucun voxel non-nul pour DBSCAN')
    db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels = db.fit_predict(pts)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    summary = {'n_clusters': int(n_clusters), 'n_noise': int(n_noise)}
    return labels, mask, summary


def compute_cluster_centroids(labels, mask, x_centers, y_centers, counts):
    """Calcule centroïdes pondérés par cluster (ignore label -1)."""
    x_centers = np.asarray(x_centers)
    y_centers = np.asarray(y_centers)
    counts = np.asarray(counts)
    Xg, Yg = np.meshgrid(x_centers, y_centers)
    pts = np.column_stack((Xg[mask], Yg[mask]))
    weights = counts[mask].astype(float)
    centroids = {}
    unique = [k for k in np.unique(labels) if k != -1]
    for k in unique:
        idx = np.where(labels == k)[0]
        if idx.size == 0:
            continue
        pts_k = pts[idx]
        w_k = weights[idx]
        wsum = w_k.sum()
        if wsum <= 0:
            cx, cy = pts_k.mean(axis=0)
        else:
            cx = float((w_k * pts_k[:,0]).sum() / wsum)
            cy = float((w_k * pts_k[:,1]).sum() / wsum)
        centroids[int(k)] = {'centroid': (cx, cy), 'weight': float(wsum), 'n_points': int(idx.size)}
    return centroids
