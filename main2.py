import os
import sys
import math
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy import fftpack
from scipy import ndimage
from skimage.transform import hough_line, hough_line_peaks

# /c:/Users/brunelgu/Documents/WorspacePython/missingGTP/main2.py
# GitHub Copilot
"""
Analyse d'un fichier data/parcel.csv contenant des points LiDAR (fid,X,Y,Z,Intensity,Classification).
- Filtre Classification == 4 (végétation)
- Affichage XY optionnel
- Voxelisation (taille par défaut 0.3) + visualisation optionnelle
- PCA pondérée sur centres de voxels -> orientation principale des rangs
- FFT 2D sur carte de densité -> orientation + espacement
- Hough sur image binaire dérivée -> orientation + espacement
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Optional skimage for Hough
try:
    SKIMAGE_AVAILABLE = True
except Exception:
    SKIMAGE_AVAILABLE = False

def load_points(csv_path):
    df = pd.read_csv(csv_path)
    # s'assurer des colonnes
    expected = ['fid', 'X', 'Y', 'Z', 'Intensity', 'Classification']
    for c in expected:
        if c not in df.columns:
            raise ValueError(f"Colonne attendue manquante: {c}")
    return df

def filter_class(df, class_value=4):
    return df[df['Classification'] == class_value].copy()

def voxelize(points_xyza, voxel_size=0.3):
    # points_xyza: ndarray Nx3 (X,Y,Z) or dataframe with X,Y,Z columns
    if isinstance(points_xyza, pd.DataFrame):
        coords = points_xyza[['X','Y','Z']].values
    else:
        coords = np.asarray(points_xyza)
    mins = coords.min(axis=0)
    idx = np.floor((coords - mins) / voxel_size).astype(int)
    # key by tuple of (ix,iy,iz)
    # aggregate counts and centroids
    dtype = [('ix', int), ('iy', int), ('iz', int)]
    keys = [tuple(row) for row in idx]
    uniq = {}
    for k, pt in zip(keys, coords):
        if k not in uniq:
            uniq[k] = {'count':0, 'sum':np.zeros(3)}
        uniq[k]['count'] += 1
        uniq[k]['sum'] += pt
    centers = []
    counts = []
    indices = []
    for (ix,iy,iz), v in uniq.items():
        cnt = v['count']
        cen = v['sum'] / cnt
        centers.append(cen)
        counts.append(cnt)
        indices.append((ix,iy,iz))
    centers = np.array(centers)
    counts = np.array(counts)
    indices = np.array(indices)
    return {
        'centers': centers,
        'counts': counts,
        'indices': indices,
        'mins': mins,
        'voxel_size': voxel_size
    }

def plot_xy(points, title="Points XY"):
    plt.figure(figsize=(6,6))
    plt.scatter(points['X'], points['Y'], s=1)
    plt.axis('equal')
    plt.title(title)
    plt.xlabel('X'); plt.ylabel('Y')
    plt.show()

def plot_voxel_centers(vox, cmap='viridis'):
    centers = vox['centers']
    counts = vox['counts']
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(centers[:,0], centers[:,1], centers[:,2], c=counts, cmap=cmap, s=10)
    ax.set_title('Centres de voxels (coloré par nombre de points)')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    plt.colorbar(sc, label='counts')
    plt.show()

def weighted_pca_2d(centers, weights):
    # centre en XY, compute weighted covariance
    pts = centers[:, :2]
    w = weights.astype(float)
    w_sum = w.sum()
    if w_sum == 0:
        raise ValueError("Poids nuls")
    mean = np.average(pts, axis=0, weights=w)
    pts_centered = pts - mean
    # weighted covariance
    cov = np.dot((pts_centered * w[:,None]).T, pts_centered) / w_sum
    eigvals, eigvecs = np.linalg.eigh(cov)
    # largest eigenvector
    idx = np.argmax(eigvals)
    principal = eigvecs[:, idx]
    angle_rad = math.atan2(principal[1], principal[0])  # direction in XY
    angle_deg = math.degrees(angle_rad) % 180  # orientation undirected
    return angle_deg, principal, eigvals

def build_density_image(vox):
    # build 2D image (X,Y) counts aggregated by ix,iy
    indices = vox['indices']  # Nx3 ints
    counts = vox['counts']
    ix = indices[:,0]; iy = indices[:,1]
    min_ix, min_iy = ix.min(), iy.min()
    ix_rel = ix - min_ix
    iy_rel = iy - min_iy
    nx = ix_rel.max() + 1
    ny = iy_rel.max() + 1
    img = np.zeros((ny, nx), dtype=float)  # rows = y
    for x, y, c in zip(ix_rel, iy_rel, counts):
        img[y, x] += c
    # flip y for display so that origin is bottom-left later if needed
    return img, (min_ix, min_iy), (nx, ny)

def fft2d_orientation_spacing(img, voxel_size):
    # Apply window to reduce edge effects
    img2 = img - img.mean()
    window = np.outer(np.hanning(img2.shape[0]), np.hanning(img2.shape[1]))
    imgw = img2 * window
    # 2D FFT
    F = np.fft.fftshift(np.fft.fft2(imgw))
    power = np.abs(F)**2
    # build frequency axes (cycles per unit length)
    ny, nx = img.shape
    fx = np.fft.fftshift(np.fft.fftfreq(nx, d=voxel_size))  # x direction
    fy = np.fft.fftshift(np.fft.fftfreq(ny, d=voxel_size))  # y direction
    FX, FY = np.meshgrid(fx, fy)
    # remove DC
    power_center = (ny//2, nx//2)
    power[power_center] = 0
    # find peak
    idx = np.unravel_index(np.argmax(power), power.shape)
    peak_fy = FY[idx]
    peak_fx = FX[idx]
    freq_mag = math.hypot(peak_fx, peak_fy)
    if freq_mag == 0:
        return None, None
    # orientation: in spatial domain, rows direction is perpendicular to frequency vector
    angle_rad = math.atan2(peak_fy, peak_fx) + math.pi/2
    angle_deg = (math.degrees(angle_rad) + 360) % 180
    spacing = 1.0 / freq_mag
    return angle_deg, spacing

def hough_orientation_spacing(img, voxel_size):
    if not SKIMAGE_AVAILABLE:
        print("skimage non disponible -> saut de Hough")
        return None, None
    # Binary image: threshold > 0
    bw = img > 0
    # compute Hough transform
    hspace, angles, dists = hough_line(bw)
    accum, angs, dists_peaks = hough_line_peaks(hspace, angles, dists, num_peaks=6)
    if len(angs) == 0:
        return None, None
    # choose the most accumulator peak
    ang = angs[0]
    # compute spacing by grouping peaks with similar angle: use dists_peaks of peaks with same angle approx
    # convert dists (in pixels) to meters
    dists_m = np.array(dists_peaks) * voxel_size
    # estimate spacing as median diff of dists corresponding to dominant angle (if multiple)
    # we consider all returned dists (filtered by +/- small angle tolerance)
    tol = np.deg2rad(2.0)
    selected = [d for a,d in zip(angs, dists_peaks) if abs(a - ang) < tol]
    if len(selected) >= 2:
        selected = np.sort(np.array(selected))
        diffs = np.diff(selected) * voxel_size
        spacing = float(np.median(diffs))
    else:
        spacing = None
    angle_deg = (math.degrees(ang) + 360) % 180
    return angle_deg, spacing

def main():
    # ----- paramètres modifiables -----
    csv_path = os.path.join('data', 'parcel.csv')
    classification_value = 4
    show_xy = True
    voxel_size = 0.3
    show_voxels = True
    # ----------------------------------
    if not os.path.exists(csv_path):
        print(f"Fichier introuvable: {csv_path}")
        sys.exit(1)
    df = load_points(csv_path)
    dfv = filter_class(df, classification_value)
    if dfv.empty:
        print("Aucun point pour la classification demandée")
        sys.exit(1)
    print(f"Points après filtre: {len(dfv)}")
    if show_xy:
        plot_xy(dfv, title=f"Points XY (Classification={classification_value})")
    vox = voxelize(dfv, voxel_size=voxel_size)
    print(f"Voxels créés: {len(vox['counts'])}")
    if show_voxels:
        plot_voxel_centers(vox)
    # PCA pondérée
    try:
        pca_angle, principal_vec, eigvals = weighted_pca_2d(vox['centers'], vox['counts'])
        print(f"PCA: orientation principale = {pca_angle:.2f}° (vecteur {principal_vec}), valeurs propres {eigvals}")
    except Exception as e:
        print("PCA échouée:", e)
        pca_angle = None
    # FFT 2D
    img2d, origin_ixiy, shape = build_density_image(vox)
    fft_angle, fft_spacing = fft2d_orientation_spacing(img2d, voxel_size)
    if fft_angle is None:
        print("FFT: pas de pic significatif trouvé")
    else:
        print(f"FFT: orientation = {fft_angle:.2f}°, espacement ≈ {fft_spacing:.3f} (m)")
        # afficher spectre si souhaité
        plt.figure(figsize=(6,5))
        plt.imshow(np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(img2d - img2d.mean())))), cmap='inferno')
        plt.title('Spectre (log power)')
        plt.colorbar(label='log power')
        plt.show()
    # Hough
    hough_angle, hough_spacing = hough_orientation_spacing(img2d, voxel_size)
    if hough_angle is None:
        print("Hough: non disponible ou aucun pic")
    else:
        print(f"Hough: orientation = {hough_angle:.2f}°, espacement ≈ {hough_spacing if hough_spacing is None else f'{hough_spacing:.3f}'} (m)")
    # Affichage de la carte de densité
    plt.figure(figsize=(6,6))
    plt.imshow(img2d, origin='lower', cmap='gray')
    plt.title('Carte de densité des voxels (Y vs X)')
    plt.colorbar(label='counts')
    plt.show()

if __name__ == '__main__':
    main()