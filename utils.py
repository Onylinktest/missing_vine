import os
import numpy as np
import pandas as pd


def load_points(path="data/parcel.csv", classification=4):
    """Read CSV and filter by Classification (case-insensitive). Returns xs, ys, zs."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    df = pd.read_csv(path)
    cols_lower = {c.lower(): c for c in df.columns}
    if "classification" not in cols_lower:
        raise ValueError("Colonne 'Classification' introuvable dans le CSV.")
    cls_col = cols_lower["classification"]
    df_f = df[df[cls_col] == classification]
    if df_f.shape[0] == 0:
        raise ValueError(f"Aucun point avec Classification == {classification}")
    if "x" in cols_lower and "y" in cols_lower:
        xcol = cols_lower["x"]; ycol = cols_lower["y"]
    elif "X" in df.columns and "Y" in df.columns:
        xcol, ycol = "X", "Y"
    else:
        possible = [c for c in df.columns if c.lower() in ("x","y")]
        if len(possible) < 2:
            raise ValueError("Colonnes X/Y introuvables.")
        xcol, ycol = possible[0], possible[1]
    xs = df_f[xcol].values
    ys = df_f[ycol].values
    zs = df_f[[c for c in df_f.columns if c.lower()=="z" or c=="Z"][0]].values if any(c.lower()=="z" or c=="Z" for c in df_f.columns) else np.zeros_like(xs)
    return xs, ys, zs


def voxelize_2d(xs, ys, voxel_size=0.3, method="count"):
    """2D histogram voxelization. Returns counts (ny,nx), x_edges, y_edges, x_centers, y_centers.

    Note: le paramètre "method" est accepté pour compatibilité, actuellement seul "count" est implémenté.
    """
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    nx = int(np.ceil((x_max - x_min) / voxel_size)) or 1
    ny = int(np.ceil((y_max - y_min) / voxel_size)) or 1
    x_edges = np.linspace(x_min, x_min + nx * voxel_size, nx+1)
    y_edges = np.linspace(y_min, y_min + ny * voxel_size, ny+1)
    counts, x_edges_out, y_edges_out = np.histogram2d(xs, ys, bins=[x_edges, y_edges])
    counts = counts.T
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    return counts, x_edges, y_edges, x_centers, y_centers
