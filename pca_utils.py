import os
import numpy as np
import pandas as pd


def pca_per_cluster(labels, mask, x_centers, y_centers, counts):
    """Calcule PCA pondérée pour chaque cluster (ignore label -1)."""
    x_centers = np.asarray(x_centers)
    y_centers = np.asarray(y_centers)
    counts = np.asarray(counts)
    Xg, Yg = np.meshgrid(x_centers, y_centers)
    pts = np.column_stack((Xg[mask], Yg[mask]))
    weights = counts[mask].astype(float)
    results = {}
    unique = [k for k in np.unique(labels) if k != -1]
    for k in unique:
        idx = np.where(labels == k)[0]
        if idx.size == 0:
            continue
        pts_k = pts[idx]
        w_k = weights[idx]
        if pts_k.shape[0] < 2:
            results[int(k)] = {'angle_deg': np.nan, 'anisotropy_ratio': np.nan, 'anisotropy_norm': np.nan, 'eigvals': None, 'eigvecs': None, 'mean': None}
            continue
        wsum = w_k.sum()
        if wsum > 0:
            mean = (w_k[:, None] * pts_k).sum(axis=0) / wsum
        else:
            mean = pts_k.mean(axis=0)
        Xc = pts_k - mean
        if wsum > 0:
            cov = (w_k[:, None, None] * (Xc[:, :, None] @ Xc[:, None, :])).sum(axis=0) / wsum
        else:
            cov = np.cov(Xc, rowvar=False)
        try:
            vals, vecs = np.linalg.eigh(cov)
        except Exception:
            results[int(k)] = {'angle_deg': np.nan, 'anisotropy_ratio': np.nan, 'anisotropy_norm': np.nan, 'eigvals': None, 'eigvecs': None, 'mean': mean}
            continue
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        lambda1 = float(vals[0])
        lambda2 = float(vals[1]) if vals.size > 1 else 0.0
        principal = vecs[:, 0]
        angle_deg = float(np.degrees(np.arctan2(principal[1], principal[0])) % 180)
        anis_ratio = float(lambda1 / lambda2) if lambda2 > 0 else np.nan
        anis_norm = float((lambda1 - lambda2) / (lambda1 + lambda2)) if (lambda1 + lambda2) != 0 else np.nan
        results[int(k)] = {'angle_deg': angle_deg, 'anisotropy_ratio': anis_ratio, 'anisotropy_norm': anis_norm, 'eigvals': vals, 'eigvecs': vecs, 'mean': mean}
    return results


def compute_principal_orientation_from_pca(pca_results, n_clusters=2):
    """KMeans on anisotropy_ratio to choose principal clusters and compute circular mean angle."""
    import numpy as _np
    labels = []
    ratios = []
    angles = {}
    for lab, info in (pca_results or {}).items():
        ar = info.get('anisotropy_ratio', _np.nan)
        ang = info.get('angle_deg', _np.nan)
        try:
            if not _np.isnan(ar) and not _np.isnan(ang):
                labels.append(int(lab))
                ratios.append(float(ar))
                angles[int(lab)] = float(ang)
        except Exception:
            continue
    if len(ratios) < 2:
        return None
    X = _np.array(ratios).reshape(-1, 1)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=min(n_clusters, X.shape[0]), random_state=0)
    assigns = kmeans.fit_predict(X)
    uniq = _np.unique(assigns)
    mean_per_group = {int(g): float(_np.mean(X[assigns == g])) for g in uniq}
    high_group = max(mean_per_group.keys(), key=lambda g: mean_per_group[g])
    selected_labels = [labels[i] for i in range(len(labels)) if assigns[i] == high_group]
    if len(selected_labels) == 0:
        return None
    angs_rad = _np.radians(_np.array([angles[l] for l in selected_labels]))
    ang2 = 2.0 * angs_rad
    s = _np.mean(_np.sin(ang2))
    c = _np.mean(_np.cos(ang2))
    mean_ang_rad = 0.5 * _np.arctan2(s, c)
    mean_ang_deg = ( _np.degrees(mean_ang_rad) ) % 180
    assignments = {int(labels[i]): int(assigns[i]) for i in range(len(labels))}
    return {'mean_angle_deg': float(mean_ang_deg), 'selected_labels': selected_labels, 'cluster_assignments': assignments}


def get_principal_orientation_angle(pca_results, n_clusters=2):
    try:
        out = compute_principal_orientation_from_pca(pca_results, n_clusters=n_clusters)
        if not out:
            return None
        return float(out.get('mean_angle_deg'))
    except Exception:
        return None
