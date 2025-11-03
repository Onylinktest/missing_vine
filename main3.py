import os
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.signal import find_peaks
from skimage.transform import hough_line, hough_line_peaks
from skimage import exposure
from sklearn.cluster import DBSCAN, KMeans

# main3.py
"""
Analyse d'un fichier data/parcel.csv contenant des points LIDAR (fid,X,Y,Z,Intensity,Classification).
- Filtre par Classification (par défaut 4 = végétation)
- Option d'affichage des points
- Voxelisation 2D (x,y) avec taille par défaut 0.3 m + affichage optionnel
- PCA pondérée sur centres de voxels -> orientation principale + anisotropie
- FFT 2D sur carte de densité -> orientation + espacement
- Hough sur image binaire dérivée des voxels -> orientation + espacement

Variables modifiables dans main (pas d'arguments CLI).
"""

import matplotlib.pyplot as plt
import json
import csv
from math import sqrt


# -------------------------
# Chargement et pré-trait.
# -------------------------
def load_points(path="data/parcel.csv", classification=4):
    """
    Lit le CSV et filtre selon la colonne 'Classification' (insensible à la casse).
    Retourne arrays x, y, z.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    df = pd.read_csv(path)
    # trouver colonne Classification quel que soit le case
    cols_lower = {c.lower(): c for c in df.columns}
    if "classification" not in cols_lower:
        raise ValueError("Colonne 'Classification' introuvable dans le CSV.")
    cls_col = cols_lower["classification"]
    df_f = df[df[cls_col] == classification]
    if df_f.shape[0] == 0:
        raise ValueError(f"Aucun point avec Classification == {classification}")
    # X/Y noms
    if "x" in cols_lower and "y" in cols_lower:
        xcol = cols_lower["x"]; ycol = cols_lower["y"]
    elif "X" in df.columns and "Y" in df.columns:
        xcol, ycol = "X", "Y"
    else:
        # essayer heuristique
        possible = [c for c in df.columns if c.lower() in ("x","y")]
        if len(possible) < 2:
            raise ValueError("Colonnes X/Y introuvables.")
        xcol, ycol = possible[0], possible[1]
    xs = df_f[xcol].values
    ys = df_f[ycol].values
    zs = df_f[[c for c in df_f.columns if c.lower()=="z" or c=="Z"][0]].values if any(c.lower()=="z" or c=="Z" for c in df_f.columns) else np.zeros_like(xs)
    return xs, ys, zs

# -------------------------
# Affichage points
# -------------------------
def plot_points(xs, ys, title="Points (Classification sélectionnée)", save_path=None, show=True):
    plt.figure(figsize=(8,8))
    plt.scatter(xs, ys, s=1, c='green', alpha=0.6)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title)
    plt.xlabel("X"); plt.ylabel("Y")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

# -------------------------
# Voxelisation 2D (histogramme)
# -------------------------
def voxelize_2d(xs, ys, voxel_size=0.3, method="count"):
    """
    Retourne:
      counts: 2D ndarray (ny, nx) avec comptages par voxel
      x_edges, y_edges: edges des bins
      x_centers, y_centers: vecteurs des centres
    """
    # np.histogram2d prend x, y (x bins = along x axis)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    nx = int(np.ceil((x_max - x_min) / voxel_size)) or 1
    ny = int(np.ceil((y_max - y_min) / voxel_size)) or 1
    # ajouter un petit buffer pour inclure max
    x_edges = np.linspace(x_min, x_min + nx * voxel_size, nx+1)
    y_edges = np.linspace(y_min, y_min + ny * voxel_size, ny+1)
    counts, x_edges_out, y_edges_out = np.histogram2d(xs, ys, bins=[x_edges, y_edges])
    # histogram2d returns shape (nx_bins, ny_bins) where first axis X; we'll transpose for (ny, nx)
    counts = counts.T  # now shape (ny, nx) corresponding to Y rows, X cols
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    return counts, x_edges, y_edges, x_centers, y_centers

def plot_voxels(counts, x_edges, y_edges, title="Voxel density", save_path=None, show=True):
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    plt.figure(figsize=(8,8))
    # amélioration contraste
    img = counts.copy()
    img = exposure.rescale_intensity(img, in_range='image', out_range=(0,255))
    plt.imshow(img[::-1, :], extent=extent, cmap='viridis', interpolation='nearest')
    plt.colorbar(label="Count (scaled)")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title)
    plt.xlabel("X"); plt.ylabel("Y")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

# -------------------------
# FFT 2D sur carte de densité
# -------------------------
def fft2_analysis(counts, voxel_size):
    """
    Analyse 2D par FFT du champ de densité.
    Retourne orientation (deg), espacement (m), et spectre (magnitude) pour debug.
    """
    # window pour réduire leakage
    img = counts.astype(float)
    if img.sum() == 0:
        raise ValueError("Image vide pour FFT.")
    # pad to next power of two for better FFT behavior (optionnel)
    ny, nx = img.shape
    # detrend / high-pass pour accentuer périodicité
    img_hp = img - ndimage.gaussian_filter(img, sigma=3)
    F = np.fft.fftshift(np.fft.fft2(img_hp))
    mag = np.abs(F)
    # coordonnées fréquences en cycles/pixel
    fx = np.fft.fftshift(np.fft.fftfreq(nx, d=1.0))
    fy = np.fft.fftshift(np.fft.fftfreq(ny, d=1.0))
    FX, FY = np.meshgrid(fx, fy)
    # supprimer composante continue centrale
    center = (ny//2, nx//2)
    mag[center] = 0
    # trouver indice du maximum du spectre
    idx_flat = np.argmax(mag)
    iy, ix = np.unravel_index(idx_flat, mag.shape)
    fpx = FX[iy, ix]
    fpy = FY[iy, ix]
    f_norm = np.hypot(fpx, fpy)
    if f_norm == 0:
        raise ValueError("Pas de pic spectral significatif trouvé.")
    # orientation spatiale: pic spectral perpendiculaire à direction des rangs -> ajouter 90 deg
    angle_rad = np.arctan2(fpy, fpx) + np.pi/2
    angle_deg = (np.degrees(angle_rad)) % 180
    # espacement en mètres: freq in cycles/pixel, pixel = voxel_size meters
    spacing_m = voxel_size / f_norm
    return float(angle_deg), float(spacing_m), mag, (FX, FY)

# -------------------------
# Hough sur image binaire dérivée des voxels
# -------------------------
def hough_analysis(counts, x_centers, y_centers, voxel_size, threshold_rel=0.2):
    """
    Binarise par seuil relatif, applique transformée de Hough pour trouver orientation principale.
    Calcule espacement en projetant points 'on' sur la normale et en détectant périodicité.
    """
    # normaliser et seuillage
    img = counts.copy().astype(float)
    if img.sum() == 0:
        raise ValueError("Image vide pour Hough.")
    img_norm = img / img.max()
    binary = img_norm >= threshold_rel
    # Hough transform
    hspace, angles, dists = hough_line(binary)
    accum, angles_peaks, dists_peaks = hough_line_peaks(hspace, angles, dists, num_peaks=6)
    if len(angles_peaks) == 0:
        raise ValueError("Aucun pic Hough trouvé.")
    # choisir angle le plus fréquent / avec plus grand accumulateur
    # angles_peaks are in radians, relative to image axis (0 is vertical? skimage: angle 0 corresponds to vertical lines)
    dominant_angle = np.median(angles_peaks)  # robust
    # Convert to spatial orientation (degrees). skimage's angle is the angle of the normal to the line wrt x axis?
    # For interpretation, compute direction vector of line: (cos(theta+pi/2), sin(theta+pi/2))
    line_dir = np.array([np.cos(dominant_angle + np.pi/2), np.sin(dominant_angle + np.pi/2)])
    orientation_deg = (np.degrees(np.arctan2(line_dir[1], line_dir[0]))) % 180

    # Espacement: récupérer positions des voxels 'on'
    rows, cols = np.nonzero(binary)
    xs = x_centers[cols]
    ys = y_centers[rows]
    pts = np.column_stack([xs, ys])
    # normale à la ligne pour projection = perpendicular to line direction
    normal = np.array([-line_dir[1], line_dir[0]])
    normal = normal / np.linalg.norm(normal)
    projections = pts @ normal  # distances along normal
    # créer histogram et détecter pics -> espacement moyen
    # trier et faire histogram
    if projections.size < 10:
        spacing = np.nan
    else:
        # histogram step about voxel_size/2
        hist_min, hist_max = projections.min(), projections.max()
        nbins = max(50, int((hist_max - hist_min) / (voxel_size/2)))
        hist, bin_edges = np.histogram(projections, bins=nbins)
        # trouver pics
        peaks, _ = find_peaks(hist, height=np.max(hist)*0.2, distance=1)
        if len(peaks) >= 2:
            peak_positions = (bin_edges[peaks] + bin_edges[peaks+1]) / 2.0
            # distances between consecutive peaks
            dists_peaks = np.diff(peak_positions)
            spacing = float(np.median(dists_peaks))
        else:
            # fallback: estimer spacing via FFT du profil 1D
            prof = ndimage.gaussian_filter1d(hist.astype(float), sigma=1)
            ft = np.abs(np.fft.rfft(prof - prof.mean()))
            freqs = np.fft.rfftfreq(len(prof), d=(bin_edges[1]-bin_edges[0]))
            if ft.size > 1:
                peak_idx = np.argmax(ft[1:]) + 1
                freq = freqs[peak_idx]
                spacing = 1.0 / freq if freq != 0 else np.nan
            else:
                spacing = np.nan
    return float(orientation_deg), float(spacing), binary

# -------------------------
# DBSCAN sur centres de voxels (ignorer counts == 0)
# -------------------------
def dbscan_on_voxels(x_centers, y_centers, counts, eps=0.6, min_samples=3, metric='euclidean'):
    """Applique DBSCAN sur les centres de voxels dont counts>0.

    eps: distance en mêmes unités que x_centers/y_centers (mètres si centres en m).
    Retourne labels array (len = number of non-zero voxels), mask (bool array same shape as counts),
    et dictionnaire summary {n_clusters, n_noise}.
    """
    x_centers = np.asarray(x_centers)
    y_centers = np.asarray(y_centers)
    counts = np.asarray(counts)

    # grilles
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


def plot_dbscan_clusters(counts, x_edges, y_edges, x_centers, y_centers, labels, mask, title='DBSCAN clusters', save_path=None, show=True):
    """Affiche la carte des voxels et superpose les centres colorés par cluster.
    labels et mask doivent correspondre (mask flatten selects the pts corresponding to labels).
    """
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    plt.figure(figsize=(8,8))
    img = counts.copy()
    img = exposure.rescale_intensity(img, in_range='image', out_range=(0,255))
    plt.imshow(img[::-1, :], extent=extent, cmap='viridis', interpolation='nearest')

    Xg, Yg = np.meshgrid(x_centers, y_centers)
    pts_all = np.column_stack((Xg[mask], Yg[mask]))

    # plot clusters: noise as black with small marker
    unique_labels = set(labels)
    colors = plt.get_cmap('tab10')
    for k in unique_labels:
        class_member_mask = (labels == k)
        xy = pts_all[class_member_mask]
        if xy.size == 0:
            continue
        if k == -1:
            plt.scatter(xy[:,0], xy[:,1], c='k', s=6, marker='x', label='noise')
        else:
            plt.scatter(xy[:,0], xy[:,1], c=[colors(k % 10)], s=8, label=f'cluster {k}')

    plt.title(title)
    plt.xlabel('X'); plt.ylabel('Y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(loc='upper right', markerscale=2, fontsize='small')
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def compute_cluster_centroids(labels, mask, x_centers, y_centers, counts):
    """Calcule les centroïdes pondérés pour chaque cluster (ignorer label -1).

    labels : array shape (n_pts_nonzero,)
    mask : bool array shape (ny, nx) indiquant les voxels utilisés
    x_centers, y_centers : arrays
    counts : array shape (ny, nx)

    Retourne dictionnaire {cluster_label: {'centroid':(x,y), 'weight':total_weight, 'n_points':n_points}}
    """
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


def plot_cluster_centroids(centroids, title='Cluster centroids', save_path=None, show=True):
    """Affiche les centroïdes sur un figure séparée (ou overlay si souhaité).
    centroids : dict retourné par compute_cluster_centroids
    """
    if not centroids:
        print('Aucun centroïde à afficher.')
        return
    plt.figure(figsize=(6,6))
    for k, info in centroids.items():
        cx, cy = info['centroid']
        w = info['weight']
        plt.scatter(cx, cy, s=50, marker='o', edgecolors='white', label=f'cluster {k} (w={w:.0f})')
        plt.annotate(str(k), (cx, cy), color='white', weight='bold', fontsize=8, ha='center', va='center')
    plt.title(title)
    plt.xlabel('X'); plt.ylabel('Y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(loc='best', fontsize='small')
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

# --- Export pour QGIS: CSV et GeoJSON ---
def export_voxels_to_csv(path, counts, x_centers, y_centers, mask=None, labels=None):
    """Export des voxels non-vides en CSV (x,y,count,cluster).
    labels doit être un array 1D correspondant aux positions True dans mask (ou None).
    """
    if mask is None:
        mask = counts > 0
    rows, cols = np.where(mask)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x','y','count','cluster'])
        for idx, (r, c) in enumerate(zip(rows, cols)):
            x = float(x_centers[c])
            y = float(y_centers[r])
            cnt = int(counts[r, c])
            cl = int(labels[idx]) if (labels is not None) else ''
            writer.writerow([x, y, cnt, cl])

def export_voxels_to_geojson(path, counts, x_edges, y_edges, x_centers, y_centers, mask=None, labels=None, as_polygons=False):
    """Export GeoJSON FeatureCollection. Par défaut exporte des points (centers).
    Si as_polygons=True, chaque voxel est exporté comme polygone à partir des edges.
    """
    if mask is None:
        mask = counts > 0
    rows, cols = np.where(mask)
    features = []
    idx = 0
    for r, c in zip(rows, cols):
        cnt = int(counts[r, c])
        props = {'count': cnt}
        if labels is not None:
            props['cluster'] = int(labels[idx])
        if as_polygons:
            # polygon from edges: (x0,y0)->(x1,y0)->(x1,y1)->(x0,y1)->(x0,y0)
            x0 = float(x_edges[c]); x1 = float(x_edges[c+1])
            y0 = float(y_edges[r]); y1 = float(y_edges[r+1])
            coords = [[ [x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0] ]]
            geom = {'type': 'Polygon', 'coordinates': coords}
        else:
            x = float(x_centers[c]); y = float(y_centers[r])
            geom = {'type': 'Point', 'coordinates': [x, y]}
        features.append({'type': 'Feature', 'geometry': geom, 'properties': props})
        idx += 1
    fc = {'type': 'FeatureCollection', 'features': features}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(fc, f, ensure_ascii=False, indent=2)


def export_centroids_to_csv(path, centroids):
    """Export des centroïdes de clusters en CSV (label,x,y,weight,n_points)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['label','x','y','weight','n_points'])
        for k, info in centroids.items():
            cx, cy = info['centroid']
            w = float(info.get('weight', 0.0))
            n = int(info.get('n_points', 0))
            writer.writerow([int(k), float(cx), float(cy), w, n])


def export_centroids_to_geojson(path, centroids):
    """Export GeoJSON des centroïdes (Point features avec attributs)."""
    features = []
    for k, info in centroids.items():
        cx, cy = info['centroid']
        props = {'label': int(k), 'weight': float(info.get('weight', 0.0)), 'n_points': int(info.get('n_points', 0))}
        geom = {'type': 'Point', 'coordinates': [float(cx), float(cy)]}
        features.append({'type': 'Feature', 'geometry': geom, 'properties': props})
    fc = {'type': 'FeatureCollection', 'features': features}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(fc, f, ensure_ascii=False, indent=2)


def _compute_alpha_shape(points, alpha):
    """Compute alpha shape (concave hull) for a set of 2D points.
    Requires shapely and scipy.spatial.Delaunay. If shapely is unavailable,
    this function will raise ImportError.
    Returns a shapely.geometry (Polygon or MultiPolygon).
    """
    # lazy imports
    try:
        from shapely.geometry import Polygon, MultiPolygon, Point
        from shapely.ops import unary_union, polygonize
    except Exception as e:
        raise ImportError("shapely is required for alpha shape computation")
    from scipy.spatial import Delaunay

    if len(points) < 4:
        # Not enough points for a polygon -> return convex hull of points
        return MultiPolygon([Point(p).buffer(0) for p in points]).convex_hull

    pts = np.array(points)
    tri = Delaunay(pts)
    triangles = pts[tri.simplices]

    def triangle_circumradius(tri_pts):
        a = np.linalg.norm(tri_pts[0] - tri_pts[1])
        b = np.linalg.norm(tri_pts[1] - tri_pts[2])
        c = np.linalg.norm(tri_pts[2] - tri_pts[0])
        s = 0.5 * (a + b + c)
        area = max(s * (s - a) * (s - b) * (s - c), 0.0)
        if area <= 0:
            return np.inf
        area = sqrt(area)
        # circumradius formula: R = (a*b*c) / (4*area_triangle)
        R = (a * b * c) / (4.0 * area)
        return R

    # Keep triangles with circumradius <= 1/alpha (common criterion). If alpha==0, keep none.
    if alpha is None or alpha <= 0:
        # treat as convex hull request
        from shapely.geometry import MultiPoint
        return MultiPoint(list(map(tuple, pts))).convex_hull

    triangles_kept = []
    for t in triangles:
        R = triangle_circumradius(t)
        if R <= 1.0 / float(alpha):
            triangles_kept.append(Polygon(t))

    if not triangles_kept:
        from shapely.geometry import MultiPoint
        return MultiPoint(list(map(tuple, pts))).convex_hull

    union = unary_union(triangles_kept)
    # polygonize the unioned triangles to produce polygons
    polys = list(polygonize(union))
    if not polys:
        return union.convex_hull
    if len(polys) == 1:
        return polys[0]
    return unary_union(polys)


def export_rows_concave_hulls(voxels_csv_path, output_geojson_path, alpha=None, min_points=3, simplify_tolerance=0.0):
    """Compute concave (alpha) hull per row_id from a voxels CSV and export GeoJSON.

    Parameters
    ----------
    voxels_csv_path: path to CSV with columns at least ['x','y','cluster'] or ['x','y','row_id']
    output_geojson_path: target GeoJSON file with polygons per row
    alpha: float or None. If None or shapely unavailable, convex hull is used.
    min_points: minimum number of points to build a polygon (else skipped)
    simplify_tolerance: if >0, simplify polygon geometry by this tolerance (meters)
    """
    # read CSV
    if not os.path.exists(voxels_csv_path):
        raise FileNotFoundError(f"Voxels CSV not found: {voxels_csv_path}")
    df = pd.read_csv(voxels_csv_path)
    # prefer existing row_id column
    if 'row_id' in df.columns:
        group_col = 'row_id'
    elif 'cluster' in df.columns:
        # fallback: try mapping cluster==label -> row_id (user should have updated file earlier)
        group_col = 'cluster'
    else:
        raise ValueError('CSV must contain column row_id or cluster')

    features = []
    # try shapely availability
    shapely_available = True
    try:
        import shapely.geometry as _sg
    except Exception:
        shapely_available = False

    for gid, g in df.groupby(group_col):
        if pd.isna(gid):
            continue
        pts = g[['x', 'y']].dropna().values
        if pts.shape[0] < min_points:
            continue
        poly_geom = None
        if shapely_available:
            try:
                poly = _compute_alpha_shape(pts, alpha)
                if simplify_tolerance and poly is not None:
                    poly = poly.simplify(simplify_tolerance)
                poly_geom = poly.__geo_interface__
            except Exception as e:
                # fallback to convex hull using scipy if alpha-shape fails
                from scipy.spatial import ConvexHull
                try:
                    hull = ConvexHull(pts)
                    hull_pts = pts[hull.vertices]
                    poly_geom = {
                        'type': 'Polygon',
                        'coordinates': [hull_pts.tolist() + [hull_pts[0].tolist()]]
                    }
                except Exception:
                    poly_geom = None
        else:
            # shapely not available -> convex hull
            from scipy.spatial import ConvexHull
            try:
                hull = ConvexHull(pts)
                hull_pts = pts[hull.vertices]
                poly_geom = {
                    'type': 'Polygon',
                    'coordinates': [hull_pts.tolist() + [hull_pts[0].tolist()]]
                }
            except Exception:
                poly_geom = None

        if poly_geom is None:
            continue

        props = {'row_id': int(gid), 'n_voxels': int(len(g))}
        features.append({'type': 'Feature', 'geometry': poly_geom, 'properties': props})

    fc = {'type': 'FeatureCollection', 'features': features}
    os.makedirs(os.path.dirname(output_geojson_path), exist_ok=True)
    with open(output_geojson_path, 'w', encoding='utf-8') as f:
        json.dump(fc, f, ensure_ascii=False, indent=2)

    return output_geojson_path

def compute_and_export_longest_lines(input_geojson, output_geojson, plot_png=None, crs=None, max_hull_vertices=2000):
    """For each polygon feature in input_geojson compute the longest straight-line
    segment between two boundary points (approximate diameter) and export as GeoJSON.

    Also optionally save a PNG overlay plotting polygons and longest lines.
    Returns path to output_geojson.
    """
    if not os.path.exists(input_geojson):
        raise FileNotFoundError(f"Input geojson not found: {input_geojson}")

    with open(input_geojson, 'r', encoding='utf-8') as f:
        fc = json.load(f)

    features_out = []
    plot_polys = []
    plot_lines = []

    for feat in fc.get('features', []):
        props = feat.get('properties', {})
        gid = props.get('row_id', props.get('label', None))
        geom = feat.get('geometry')
        if geom is None:
            continue
        typ = geom.get('type')
        coords = None
        # handle Polygon or MultiPolygon
        if typ == 'Polygon':
            ring = geom.get('coordinates', [[]])[0]
            coords = np.array(ring)
        elif typ == 'MultiPolygon':
            # choose largest polygon by area (approx via shoelace)
            best = None
            best_area = -1
            for poly in geom.get('coordinates', []):
                ring = np.array(poly[0])
                if ring.shape[0] < 3:
                    continue
                # polygon area via shoelace
                x = ring[:,0]; y = ring[:,1]
                area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
                if area > best_area:
                    best_area = area
                    best = ring
            coords = best
        else:
            # skip non-polygon geometries
            continue

        if coords is None or coords.shape[0] < 2:
            continue

        # reduce candidate points using convex hull of boundary if too many points
        pts = coords.copy()
        try:
            if pts.shape[0] > max_hull_vertices:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(pts)
                pts_cand = pts[hull.vertices]
            else:
                pts_cand = pts
        except Exception:
            pts_cand = pts

        # compute pairwise distances efficiently - if too many points, operate on candidates
        n = pts_cand.shape[0]
        if n < 2:
            continue
        if n < 3000:
            # compute full distance matrix
            dx = pts_cand[:, None, 0] - pts_cand[None, :, 0]
            dy = pts_cand[:, None, 1] - pts_cand[None, :, 1]
            dist2 = dx*dx + dy*dy
            i, j = np.unravel_index(np.argmax(dist2), dist2.shape)
            p1 = pts_cand[int(i)]; p2 = pts_cand[int(j)]
        else:
            # approximate: sample subset or use farthest point heuristic
            # use pairwise between a random sample and all points
            rng = np.random.default_rng(0)
            sample_idx = rng.choice(n, size=min(500, n), replace=False)
            sub = pts_cand[sample_idx]
            # distances between sub and pts_cand
            from scipy.spatial import distance_matrix
            D = distance_matrix(sub, pts_cand)
            idx = np.unravel_index(np.argmax(D), D.shape)
            p1 = sub[int(idx[0])]; p2 = pts_cand[int(idx[1])]

        length = float(np.linalg.norm(p2 - p1))
        line_geom = {'type': 'LineString', 'coordinates': [[float(p1[0]), float(p1[1])], [float(p2[0]), float(p2[1])]]}

        feat_line = {'type': 'Feature', 'geometry': line_geom, 'properties': {'row_id': int(gid) if gid is not None else None, 'max_length_m': length}}
        features_out.append(feat_line)

        # store for plotting
        plot_polys.append(coords)
        plot_lines.append((p1, p2))

    out_fc = {'type': 'FeatureCollection', 'features': features_out}
    if crs:
        try:
            out_fc['crs'] = {'type': 'name', 'properties': {'name': str(crs)}}
        except Exception:
            pass
    os.makedirs(os.path.dirname(output_geojson), exist_ok=True)
    with open(output_geojson, 'w', encoding='utf-8') as f:
        json.dump(out_fc, f, ensure_ascii=False, indent=2)

    # plot if requested
    if plot_png:
        plt.figure(figsize=(8,8))
        for poly in plot_polys:
            xs = poly[:,0]; ys = poly[:,1]
            plt.fill(xs, ys, edgecolor='black', facecolor='none', linewidth=0.8)
        for (p1,p2) in plot_lines:
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], '-r', linewidth=1.5)
        # draw crosses at endpoints for visibility
        try:
            for (p1,p2) in plot_lines:
                plt.plot([p1[0]], [p1[1]], marker='x', color='blue', markersize=6, markeredgewidth=1.5)
                plt.plot([p2[0]], [p2[1]], marker='x', color='blue', markersize=6, markeredgewidth=1.5)
        except Exception:
            pass
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title('Longest lines per row (max segment)')
        os.makedirs(os.path.dirname(plot_png), exist_ok=True)
        plt.savefig(plot_png, dpi=150, bbox_inches='tight')
        plt.close()

    return output_geojson

def pca_per_cluster(labels, mask, x_centers, y_centers, counts):
    """Calcule PCA pondérée pour chaque cluster (ignore label -1).

    Retourne dict {label: {'angle_deg':..., 'anisotropy_ratio':..., 'anisotropy_norm':..., 'eigvals':..., 'eigvecs':..., 'mean':(...) }}
    """
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


def plot_cluster_pca_axes(pca_results, scale=1.0, title='PCA axes per cluster', save_path=None, show=True):
    """Trace un plot des axes principaux pour chaque cluster donné dans pca_results (dict).
    Chaque entrée doit contenir 'mean' and 'eigvecs' and 'eigvals'.
    """
    if not pca_results:
        print('Aucun résultat PCA à afficher.')
        return
    plt.figure(figsize=(8,8))
    for k, info in pca_results.items():
        mean = info.get('mean')
        vecs = info.get('eigvecs')
        vals = info.get('eigvals')
        if mean is None or vecs is None or vals is None:
            continue
        cx, cy = float(mean[0]), float(mean[1])
        principal = vecs[:,0]
        # longueur representative -> scale * sqrt(lambda1)
        length = scale * (np.sqrt(vals[0]) if vals[0] > 0 else 1.0)
        x0, y0 = cx - principal[0]*length, cy - principal[1]*length
        x1, y1 = cx + principal[0]*length, cy + principal[1]*length
        plt.plot([x0, x1], [y0, y1], '-r', linewidth=2)
        plt.scatter([cx], [cy], c='white', s=30, edgecolors='k')
        plt.text(cx, cy, str(k), color='black', fontsize=8, ha='center', va='center')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title)
    plt.xlabel('X'); plt.ylabel('Y')
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

def compute_principal_orientation_from_pca(pca_results, n_clusters=2):
    """
    Détermine l'orientation principale à partir des résultats PCA par cluster.
    - Utilise KMeans (n_clusters=2) sur la feature 'anisotropy_ratio' pour séparer clusters "fortement anisotropes"
      et "faiblement anisotropes".
    - Sélectionne le groupe dont la moyenne d'anisotropy_ratio est la plus élevée.
    - Calcule la moyenne circulaire des angles (traitant la périodicité 180°) pour les clusters sélectionnés.

    Retourne None si pas assez de données valides, sinon dict {
      'mean_angle_deg': float, 'selected_labels': [int,...], 'cluster_assignments': {label:cluster_index,...}
    }
    """
    import numpy as _np
    # collecter ratios valides
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
        # pas assez de clusters valides pour KMeans
        return None

    X = _np.array(ratios).reshape(-1, 1)
    # KMeans 1D
    kmeans = KMeans(n_clusters=min(n_clusters, X.shape[0]), random_state=0)
    assigns = kmeans.fit_predict(X)

    # choisir le groupe avec la plus grande moyenne d'anisotropy_ratio
    uniq = _np.unique(assigns)
    mean_per_group = {int(g): float(_np.mean(X[assigns == g])) for g in uniq}
    high_group = max(mean_per_group.keys(), key=lambda g: mean_per_group[g])

    selected_labels = [labels[i] for i in range(len(labels)) if assigns[i] == high_group]

    if len(selected_labels) == 0:
        return None

    # calcul de la moyenne circulaire (periodicité 180° -> double angles)
    angs_rad = _np.radians(_np.array([angles[l] for l in selected_labels]))
    ang2 = 2.0 * angs_rad
    s = _np.mean(_np.sin(ang2))
    c = _np.mean(_np.cos(ang2))
    mean_ang_rad = 0.5 * _np.arctan2(s, c)
    mean_ang_deg = ( _np.degrees(mean_ang_rad) ) % 180

    assignments = {int(labels[i]): int(assigns[i]) for i in range(len(labels))}

    return {'mean_angle_deg': float(mean_ang_deg), 'selected_labels': selected_labels, 'cluster_assignments': assignments}

# helper: renvoie uniquement l'angle principal provenant de la PCA par cluster
def get_principal_orientation_angle(pca_results, n_clusters=2):
    """Retourne l'angle moyen (deg) déterminé à partir des résultats PCA par cluster, ou None."""
    try:
        out = compute_principal_orientation_from_pca(pca_results, n_clusters=n_clusters)
        if not out:
            return None
        return float(out.get('mean_angle_deg'))
    except Exception:
        return None

#--

# -------------------------
# MAIN (variables à modifier ici)
# -------------------------
def main():
    # ---- paramètres modifiables ----
    csv_path = "data/parcel.csv"
    classification = 4  # la classe à analyser
    voxel_size = 0.3    # taille du voxel (m)

    # affichages
    show_points = False
    save_points_plot = False
    show_voxels = False
    save_voxels_plot = False
    show_hough_binary = False
    save_hough_plot = False
    show_dbscan = False
    save_dbscan_plot = False
    show_cluster_centroids = False
    save_cluster_centroids_plot = False
    show_pca_axes = False
    save_pca_axes_plot = False
    # hulls export
    compute_row_hulls = True
    hull_alpha = 0.1
    hull_min_points = 8
    hull_simplify = 0

    # paramètre Hough (seuil relatif)
    hough_threshold_rel = 0.25

    # dossier de sortie pour sauvegarder les plots
    output_dir = "output"

    # parametres parcelle (modifiable ici)
    parcel_density = 4000        # densité (cep/ha) par défaut
    parcel_intercep = None       # intercep (m) si fourni par l'utilisateur, sinon calculé
    # --------------------------------

    print("Chargement des points...")
    xs, ys, zs = load_points(csv_path, classification=classification)
    print(f"Points chargés: {len(xs)} (classification={classification})")

    if show_points or save_points_plot:
        pp_path = os.path.join(output_dir, 'points.png') if save_points_plot else None
        plot_points(xs, ys, title=f"Points classification {classification}", save_path=pp_path, show=show_points)

    print("Voxelisation 2D...")
    counts, x_edges, y_edges, x_centers, y_centers = voxelize_2d(xs, ys, voxel_size=voxel_size)
    print(f"Voxels: {counts.shape[1]} x {counts.shape[0]} (nx x ny)")

    if show_voxels or save_voxels_plot:
        vp_path = os.path.join(output_dir, f'voxels_vs_{voxel_size:.2f}m.png') if save_voxels_plot else None
        plot_voxels(counts, x_edges, y_edges, title=f"Voxel density (voxel_size={voxel_size} m)", save_path=vp_path, show=show_voxels)

    print("Analyse FFT 2D sur carte de densité...")
    try:
        fft_angle, fft_spacing, fft_mag, fxfy = fft2_analysis(counts, voxel_size)
        print(f"FFT orientation: {fft_angle:.2f} deg, espacement: {fft_spacing:.3f} m")
    except Exception as e:
        print("FFT failed:", e)
        fft_angle, fft_spacing = None, None

    print("Analyse Hough sur image binaire...")
    try:
        hough_angle, hough_spacing, binary = hough_analysis(counts, x_centers, y_centers, voxel_size, threshold_rel=hough_threshold_rel)
        print(f"Hough orientation: {hough_angle:.2f} deg, espacement: {hough_spacing:.3f} m")
        if show_voxels or save_hough_plot:
            # afficher image binaire avec lignes approximatives
            extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
            hb_path = os.path.join(output_dir, 'hough_binary.png') if save_hough_plot else None
            plt.figure(figsize=(8,8))
            plt.imshow(binary[::-1, :], extent=extent, cmap='gray')
            plt.title("Image binaire utilisée pour Hough")
            plt.gca().set_aspect('equal', adjustable='box')
            if hb_path:
                os.makedirs(os.path.dirname(hb_path), exist_ok=True)
                plt.savefig(hb_path, dpi=150, bbox_inches='tight')
            if show_hough_binary:
                plt.show()
            else:
                plt.close()
    except Exception as e:
        print("Hough failed:", e)
        hough_angle, hough_spacing = None, None

    # Toujours exécuter DBSCAN (nécessaire pour la suite). Affichage/sauvegarde conditionnés par les flags.
    print("DBSCAN sur centres de voxels...")
    try:
        dbscan_labels, dbscan_mask, dbscan_summary = dbscan_on_voxels(x_centers, y_centers, counts)
        print(f"DBSCAN clusters: {dbscan_summary['n_clusters']}, noise: {dbscan_summary['n_noise']}")
        if show_dbscan or save_dbscan_plot:
            dbs_path = os.path.join(output_dir, 'dbscan_clusters.png') if save_dbscan_plot else None
            plot_dbscan_clusters(counts, x_edges, y_edges, x_centers, y_centers, dbscan_labels, dbscan_mask, save_path=dbs_path, show=show_dbscan)
    except Exception as e:
        print("DBSCAN failed:", e)
        # définir des valeurs par défaut pour que la suite du script n'échoue pas
        dbscan_labels, dbscan_mask, dbscan_summary = None, None, {'n_clusters': 0, 'n_noise': 0}

    # Calculer et afficher les centroïdes des clusters DBSCAN
    try:
        if 'dbscan_labels' in locals() and 'dbscan_mask' in locals():
            centroids = compute_cluster_centroids(dbscan_labels, dbscan_mask, x_centers, y_centers, counts)
            cpath = os.path.join(output_dir, 'cluster_centroids.png') if save_cluster_centroids_plot else None
            plot_cluster_centroids(centroids, title="Centroids des clusters DBSCAN", save_path=cpath, show=show_cluster_centroids)
        else:
            print('Pas de résultat DBSCAN disponible pour calculer les centroïdes.')
    except Exception as e:
        print("Erreur lors du calcul ou de l'affichage des centroïdes:", e)

    # Calculer et afficher PCA par cluster (mode silencieux)
    try:
        print('Lancement PCA par cluster...')
        pca_results = pca_per_cluster(dbscan_labels, dbscan_mask, x_centers, y_centers, counts)
    except Exception as e:
        print('PCA par cluster failed:', e)
        pca_results = None

    # Affichage / sauvegarde des axes PCA (sans impressions intermédiaires)
    if pca_results:
        if 'pca_results' in locals() and pca_results:
            pca_path = os.path.join(output_dir, 'pca_axes_clusters.png') if save_pca_axes_plot else None
            plot_cluster_pca_axes(pca_results, scale=0.5, title="PCA axes par cluster DBSCAN", save_path=pca_path, show=show_pca_axes)
    else:
        print('Aucun résultat PCA par cluster à afficher.')

    # Calcul et affichage succinct de l'orientation principale issue de la PCA
    print('Calcul de l\'orientation principale à partir des résultats PCA par cluster...')
    principal_angle = get_principal_orientation_angle(pca_results, n_clusters=2)
    if principal_angle is not None:
        print(f"Orientation principale (PCA clusters): {principal_angle:.2f} deg")
    else:
        print('Orientation principale (PCA clusters): N/A')
    
    # Résumé
    print("\n--- Résumé des orientations et espacements ---")
    pca_str = f"{principal_angle:.2f} deg" if (principal_angle is not None) else 'N/A'
    fa = f"{fft_angle:.2f} deg" if (fft_angle is not None) else 'N/A'
    fs = f"{fft_spacing:.3f} m" if (fft_spacing is not None) else 'N/A'
    ha = f"{hough_angle:.2f} deg" if (hough_angle is not None) else 'N/A'
    hs = f"{hough_spacing:.3f} m" if (hough_spacing is not None) else 'N/A'

    print(f"PCA:    orientation={pca_str}")
    print(f"FFT:    orientation={fa}, espacement={fs}")
    print(f"Hough:  orientation={ha}, espacement={hs}")

    # Calcul des paramètres de parcelle (interrang via FFT, intercep calculé si absent)
    interrang = None
    if 'fft_spacing' in locals() and fft_spacing is not None and not (isinstance(fft_spacing, float) and np.isnan(fft_spacing)):
        interrang = float(fft_spacing)
    # fallback option si FFT non disponible
    if interrang is None and 'hough_spacing' in locals() and hough_spacing is not None and not (isinstance(hough_spacing, float) and np.isnan(hough_spacing)):
        interrang = float(hough_spacing)

    # calculer intercep si non fourni: density = 10000 / (interrang * intercep) -> intercep = 10000 / (density * interrang)
    intercep = None
    if parcel_intercep is not None:
        intercep = float(parcel_intercep)
    else:
        if interrang is not None and parcel_density is not None and parcel_density > 0:
            try:
                intercep = float(10000.0 / (float(parcel_density) * round(interrang, 1)))
            except Exception:
                intercep = None

    # conserver une version arrondie pour l'affichage (1 chiffre après la virgule)
    interrang_display = round(interrang, 1) if interrang is not None else None
    intercep_display = round(intercep, 1) if intercep is not None else None

    # afficher résumé de la parcelle
    print("\n--- Résumé de la parcelle ---")
    print(f"Densité (cep/ha): {parcel_density}")
    if interrang_display is not None:
        print(f"Interrang (m): {interrang_display:.1f} m")
    else:
        print("Interrang (m): N/A")
    print(f"Intercep (m): {intercep_display:.1f} m" if intercep_display is not None else "Intercep (m): N/A")

    # Exporter les résultats pour QGIS (CSV + GeoJSON)
    save_qgis_exports = True  # flag pour activer/désactiver les exports
    if save_qgis_exports:
        print("\nExportation des résultats pour QGIS...")
        try:
            # Voxels non-vides: export CSV et GeoJSON
            export_voxels_to_csv(os.path.join(output_dir, 'voxels_non_vides.csv'), counts, x_centers, y_centers, mask=dbscan_mask, labels=dbscan_labels)
            export_voxels_to_geojson(os.path.join(output_dir, 'voxels_non_vides.geojson'), counts, x_edges, y_edges, x_centers, y_centers, mask=dbscan_mask, labels=dbscan_labels, as_polygons=False)
            print("Voxels non-vides exportés (CSV + GeoJSON).")
        except Exception as e:
            print("Erreur lors de l'export des voxels:", e)

        try:
            # Centroïdes des clusters: export CSV et GeoJSON
            export_centroids_to_csv(os.path.join(output_dir, 'centroids_clusters.csv'), centroids)
            export_centroids_to_geojson(os.path.join(output_dir, 'centroids_clusters.geojson'), centroids)
            print("Centroïdes des clusters exportés (CSV + GeoJSON).")
        except Exception as e:
            print("Erreur lors de l'export des centroïdes:", e)

        # Optionnel: exporter tous les voxels (y compris vides) pour vérification
        try:
            all_voxels_counts, _, _, _, _ = voxelize_2d(xs, ys, voxel_size=voxel_size, method="count")
            export_voxels_to_csv(os.path.join(output_dir, 'voxels_tous.csv'), all_voxels_counts, x_centers, y_centers)
            print("Tous les voxels exportés (CSV).")
        except Exception as e:
            print("Erreur lors de l'export de tous les voxels:", e)

        print("Exports terminés.")

        # Recomposition des rangs à partir des centroïdes (export CSV + GeoJSON)
        try:
            # choisir espacement: preferer Hough puis FFT
            spacing_use = None
            if 'fft_spacing' in locals() and fft_spacing is not None and not (isinstance(fft_spacing, float) and np.isnan(fft_spacing)):
                spacing_use = fft_spacing
            elif 'hough_spacing' in locals() and hough_spacing is not None and not (isinstance(hough_spacing, float) and np.isnan(hough_spacing)):
                spacing_use = hough_spacing


            if 'centroids' in locals() and centroids and spacing_use is not None and principal_angle is not None:
                out_csv = os.path.join(output_dir, 'centroids_rows.csv')
                out_geo = os.path.join(output_dir, 'centroids_rows.geojson')
                df_rows = recompose_rows_from_centroids(centroids, principal_angle, spacing_use, tolerance=0.5, output_csv=out_csv, output_geojson=out_geo, bins_frac=50)
                print(f"Recomposition terminée. Résultats sauvés: {out_csv}, {out_geo}")
            else:
                print('Recomposition des rangs non effectuée (centroids, spacing ou orientation manquants).')
        except Exception as e:
            print('Erreur lors de la recomposition des rangs:', e)

        # --- Mettre à jour les fichiers voxels_non_vides (CSV + GeoJSON) en ajoutant row_id ---
        try:
            # construire mapping label -> row_id depuis df_rows si présent, sinon tenter de lire le CSV de sortie
            rows_map = {}
            if 'df_rows' in locals() and df_rows is not None and not df_rows.empty:
                try:
                    rows_map = df_rows.set_index('label')['row_id'].to_dict()
                except Exception:
                    rows_map = {}
            else:
                # lire le CSV écrit par recompose_rows_from_centroids si disponible
                try:
                    tmp = pd.read_csv(os.path.join(output_dir, 'centroids_rows.csv'))
                    if 'label' in tmp.columns and 'row_id' in tmp.columns:
                        rows_map = tmp.set_index('label')['row_id'].to_dict()
                except Exception:
                    rows_map = {}

            vox_csv_path = os.path.join(output_dir, 'voxels_non_vides.csv')
            vox_geo_path = os.path.join(output_dir, 'voxels_non_vides.geojson')

            # Mettre à jour CSV si présent
            if os.path.exists(vox_csv_path):
                try:
                    df_vox_non = pd.read_csv(vox_csv_path)
                    # la colonne 'cluster' peut être vide/chaine; forcer en numérique
                    if 'cluster' in df_vox_non.columns:
                        df_vox_non['cluster'] = pd.to_numeric(df_vox_non['cluster'], errors='coerce')
                        # mappe avec valeurs manquantes pour clusters inconnus
                        df_vox_non['row_id'] = df_vox_non['cluster'].map(rows_map).astype('Int64')
                    else:
                        df_vox_non['row_id'] = pd.NA
                    df_vox_non.to_csv(vox_csv_path, index=False)
                    print(f"Mis à jour: {vox_csv_path} (colonne row_id ajoutée)")
                except Exception as e:
                    print('Erreur mise à jour CSV voxels_non_vides:', e)

            # Mettre à jour GeoJSON si présent
            if os.path.exists(vox_geo_path):
                try:
                    with open(vox_geo_path, 'r', encoding='utf-8') as f:
                        fc = json.load(f)
                    for feat in fc.get('features', []):
                        props = feat.get('properties', {})
                        cl = props.get('cluster', None)
                        rid = None
                        try:
                            if cl is not None:
                                rid = rows_map.get(int(cl))
                        except Exception:
                            rid = None
                        props['row_id'] = int(rid) if rid is not None else None
                        feat['properties'] = props
                    with open(vox_geo_path, 'w', encoding='utf-8') as f:
                        json.dump(fc, f, ensure_ascii=False, indent=2)
                    print(f"Mis à jour: {vox_geo_path} (propriété row_id ajoutée)")
                except Exception as e:
                    print('Erreur mise à jour GeoJSON voxels_non_vides:', e)
        except Exception as e:
            print('Erreur lors de la mise à jour des voxels_non_vides avec row_id:', e)

        # --- Exporter enveloppes (concave/alpha-shape) par rang -> GeoJSON (QGIS) ---
        try:
            if compute_row_hulls:
                vox_csv_path = os.path.join(output_dir, 'voxels_non_vides.csv')
                rows_hulls_out = os.path.join(output_dir, 'rows_hulls.geojson')
                if os.path.exists(vox_csv_path):
                    try:
                        export_rows_concave_hulls(vox_csv_path, rows_hulls_out, alpha=hull_alpha, min_points=hull_min_points, simplify_tolerance=hull_simplify)
                        print(f"Enveloppes des rangs exportées: {rows_hulls_out}")
                        # calculer aussi la ligne maximale pour chaque shape et sauvegarder
                        try:
                            rows_lines_out = os.path.join(output_dir, 'rows_hulls_max_lines.geojson')
                            rows_lines_png = os.path.join(output_dir, 'rows_hulls_max_lines.png')
                            compute_and_export_longest_lines(rows_hulls_out, rows_lines_out, plot_png=rows_lines_png)
                            print(f"Lignes maximales par rang exportées: {rows_lines_out} (plot: {rows_lines_png})")
                            # Recreate cep modelling: compute theoretical positions along longest line per row
                            try:
                                rows_lines_in = rows_lines_out
                                ceps_csv = os.path.join(output_dir, 'ceps_model.csv')
                                ceps_geo = os.path.join(output_dir, 'ceps_model.geojson')
                                intercep_use = None
                                try:
                                    intercep_use = float(intercep) if intercep is not None else None
                                except Exception:
                                    intercep_use = None

                                if intercep_use is not None and os.path.exists(rows_lines_in):
                                    try:
                                        df_ceps = generate_and_export_ceps(rows_lines_in, intercep_use,
                                                                          existing_points=np.column_stack((xs, ys)),
                                                                          presence_tol=0.5, ambiguous_tol=1.5,
                                                                          output_csv=ceps_csv, output_geojson=ceps_geo)
                                        print(f"Modèle des ceps exporté: {ceps_csv}, {ceps_geo}")
                                    except Exception as e:
                                        print('Erreur lors de la modélisation des ceps:', e)
                                else:
                                    print('Modélisation des ceps non exécutée (intercep absent ou rows_lines manquant).')
                            except Exception as e:
                                print('Erreur globale modélisation ceps:', e)
                        except Exception as e:
                            print('Erreur calcul lignes maximales par rang:', e)
                    except Exception as e:
                        print('Erreur lors du calcul/export des enveloppes des rangs:', e)
                else:
                    print(f"CSV voxels non-vides introuvable pour hulls: {vox_csv_path}")
        except Exception as e:
            print('Erreur globale lors de l\'export des enveloppes des rangs:', e)

def recompose_rows_from_centroids(centroids, theta_deg, spacing_m, tolerance=0.4, output_csv=None, output_geojson=None, bins_frac=50):
    """
    Regroupe/numérote des rangées à partir des centroides de clusters.

    Paramètres
    ----------
    centroids : dict ou pandas.DataFrame
        Si dict, format attendu {label: {'centroid':(x,y), ...}, ...}.
        Si DataFrame, doit contenir colonnes ['label','x','y'] ou ['x','y'].
    theta_deg : float
        Orientation principale en degrés (0-180). Les rangées sont supposées
        orientées selon cet angle; on projette sur l'axe orthogonal pour numéroter.
    spacing_m : float
        Espacement moyen entre rangs (m). Doit être > 0.
    tolerance : float (0..1)
        Fraction de spacing_m au‑delà de laquelle un centroïde est rejeté
        comme non aligné sur une rangée.
    output_csv : str or None
        Chemin facultatif pour sauvegarder un CSV des centroïdes avec colonne 'row_id'.
    output_geojson : str or None
        Chemin facultatif pour sauvegarder un GeoJSON des centroïdes avec propriété 'row_id'.
    bins_frac : int
        Nombre de bins pour estimer le décalage (phase) via histogramme sur la fraction
        de projection modulo spacing.

    Retourne
    -------
    pandas.DataFrame
        DataFrame contenant au moins les colonnes ['label','x','y','u_proj','row_id','residual_m','accepted','delta_m'].

    Notes
    -----
    - La méthode estime automatiquement le décalage delta dans [0, spacing_m) en
      recherchant la phase la plus fréquent dans la distribution des projections
      modulo l'espacement (max de l'histogramme). La fonction est vectorisée.
    """
    # vérifications basiques
    if spacing_m is None or spacing_m <= 0:
        raise ValueError('spacing_m doit être un nombre positif.')

    # préparer DataFrame depuis centroids (flexible)
    if isinstance(centroids, pd.DataFrame):
        df = centroids.copy()
    else:
        # dict attendu
        rows = []
        for k, info in (centroids or {}).items():
            cx, cy = info.get('centroid', (None, None))
            rows.append({'label': int(k), 'x': float(cx), 'y': float(cy), 'weight': float(info.get('weight', 0.0)), 'n_points': int(info.get('n_points', 0))})
        df = pd.DataFrame(rows)

    if df.shape[0] == 0:
        print('Aucun centroïde fourni à recompose_rows_from_centroids.')
        return df

    if 'label' not in df.columns:
        df['label'] = np.arange(len(df))
    # for safety ensure x,y columns exist
    if not {'x','y'}.issubset(df.columns):
        raise ValueError("DataFrame de centroïdes doit contenir les colonnes 'x' et 'y'.")

    # convertir en numpy
    pts = df[['x','y']].to_numpy(dtype=float)

    # vecteur direction des rangs et normale (projection axis)
    theta_rad = np.deg2rad(float(theta_deg))
    dir_vec = np.array([np.cos(theta_rad), np.sin(theta_rad)])
    # axe orthogonal pour numéroter les rangs
    normal = np.array([-dir_vec[1], dir_vec[0]])
    normal = normal / np.linalg.norm(normal)

    # projections scalaires (m) sur l'axe normal
    u = pts @ normal  # shape (N,)

    # trouver la fraction modulo spacing_m dans [0,1)
    u_scaled = u / float(spacing_m)
    frac = u_scaled - np.floor(u_scaled)

    # estimer le décalage delta via histogramme des fractions
    bins = np.linspace(0.0, 1.0, int(bins_frac) + 1)
    hist, edges = np.histogram(frac, bins=bins)
    max_idx = int(np.argmax(hist))
    delta_frac = 0.5 * (edges[max_idx] + edges[max_idx + 1])
    delta_m = float(delta_frac * spacing_m)

    # assigner row_id
    row_id = np.rint((u - delta_m) / spacing_m).astype(int)

    # residu (distance en m au centre du rang assigné)
    residual = (u - delta_m) - row_id * spacing_m
    residual_abs = np.abs(residual)

    # accepted mask
    tol_m = float(tolerance) * spacing_m
    accepted = residual_abs <= tol_m

    # construire DataFrame résultat
    df_out = df.copy()
    df_out['u_proj_m'] = u
    df_out['row_id'] = row_id
    df_out['residual_m'] = residual
    df_out['residual_abs_m'] = residual_abs
    df_out['accepted'] = accepted
    df_out['delta_m'] = delta_m

    # logs concis
    n_total = len(df_out)
    n_accept = int(df_out['accepted'].sum())
    print(f"Recomposition rangs: {n_accept}/{n_total} centroïdes acceptés (tolerance={tolerance}*spacing).")
    print(f"Estimated delta = {delta_m:.3f} m (fraction bin {delta_frac:.3f}).")

    # sauvegardes optionnelles
    if output_csv:
        try:
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            df_out.to_csv(output_csv, index=False)
            print(f"Centroids avec row_id exportés en CSV: {output_csv}")
        except Exception as e:
            print("Erreur lors de la sauvegarde CSV:", e)

    if output_geojson:
        try:
            features = []
            for _, r in df_out.iterrows():
                props = {'label': int(r['label']), 'row_id': int(r['row_id']) if pd.notna(r['row_id']) else None, 'accepted': bool(r['accepted']), 'residual_m': float(r['residual_m'])}
                geom = {'type': 'Point', 'coordinates': [float(r['x']), float(r['y'])]}
                features.append({'type': 'Feature', 'geometry': geom, 'properties': props})
            fc = {'type': 'FeatureCollection', 'features': features}
            os.makedirs(os.path.dirname(output_geojson), exist_ok=True)
            with open(output_geojson, 'w', encoding='utf-8') as f:
                json.dump(fc, f, ensure_ascii=False, indent=2)
            print(f"Centroids avec row_id exportés en GeoJSON: {output_geojson}")
        except Exception as e:
            print("Erreur lors de la sauvegarde GeoJSON:", e)

    return df_out


def generate_and_export_ceps(rows_lines_geojson, s, existing_points=None,
                             presence_tol=0.5, ambiguous_tol=1.5,
                             output_csv=None, output_geojson=None, crs=None):
    """Model ceps along each line feature in a lines GeoJSON.

    Parameters
    ----------
    rows_lines_geojson : str
        Path to GeoJSON with LineString features per row (properties must include 'row_id').
    s : float
        Nominal inter-cep spacing in meters.
    existing_points : array-like Nx2, optional
        Measured points to compare against for presence/ambiguity detection.
    presence_tol, ambiguous_tol : float
        Distance thresholds (m) for status assignment.
    output_csv, output_geojson : str
        Output paths to save modeled ceps.

    Returns
    -------
    pandas.DataFrame
        Modeled ceps with columns ['row_id','seq','x','y','status','nearest_dist_m']
    """
    if not os.path.exists(rows_lines_geojson):
        raise FileNotFoundError(f"Rows lines GeoJSON not found: {rows_lines_geojson}")
    if s is None or s <= 0:
        raise ValueError('spacing s must be positive')

    with open(rows_lines_geojson, 'r', encoding='utf-8') as f:
        fc = json.load(f)

    # prepare KDTree if possible
    ep = None
    tree = None
    try:
        if existing_points is not None:
            ep = np.asarray(existing_points, dtype=float)
            if ep.ndim == 1 and ep.size == 2:
                ep = ep.reshape(1, 2)
            from scipy.spatial import cKDTree
            if ep.size > 0:
                tree = cKDTree(ep)
    except Exception:
        tree = None
        if existing_points is not None:
            ep = np.asarray(existing_points, dtype=float)

    records = []

    for feat in fc.get('features', []):
        props = feat.get('properties', {})
        rid = props.get('row_id', props.get('label', None))
        geom = feat.get('geometry')
        if geom is None:
            continue
        typ = geom.get('type')
        if typ != 'LineString':
            # skip non-lines
            continue
        coords = geom.get('coordinates', [])
        if len(coords) < 2:
            continue
        p1 = np.array(coords[0], dtype=float)
        p2 = np.array(coords[-1], dtype=float)
        vec = p2 - p1
        L = float(np.linalg.norm(vec))
        if L <= 0:
            continue

        n = int(np.floor(L / float(s)))
        if n <= 0:
            continue
        r = L - n * float(s)
        m = r / 2.0

        if n == 1:
            dists = np.array([L / 2.0])
        else:
            # positions are evenly distributed between start+m and end-m (inclusive)
            dists = np.linspace(m, L - m, n)

        for seq, d in enumerate(dists):
            frac = d / L
            pt = p1 + frac * vec
            nearest = None
            status = 'missing'
            if tree is not None:
                try:
                    dd, _ = tree.query(pt, k=1)
                    nearest = float(dd)
                except Exception:
                    nearest = None
            elif ep is not None and ep.size > 0:
                dd = np.hypot(ep[:, 0] - pt[0], ep[:, 1] - pt[1])
                nearest = float(np.min(dd))

            if nearest is None:
                status = 'unknown'
            else:
                if nearest <= presence_tol:
                    status = 'present'
                elif nearest <= ambiguous_tol:
                    status = 'ambiguous'
                else:
                    status = 'missing'

            records.append({'row_id': int(rid) if rid is not None else None,
                            'seq': int(seq), 'x': float(pt[0]), 'y': float(pt[1]),
                            'status': status, 'nearest_dist_m': nearest})

    df_ceps = pd.DataFrame.from_records(records, columns=['row_id', 'seq', 'x', 'y', 'status', 'nearest_dist_m'])

    if output_csv:
        try:
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            df_ceps.to_csv(output_csv, index=False)
        except Exception as e:
            print('Erreur export CSV ceps:', e)

    if output_geojson:
        try:
            feats_out = []
            for _, r in df_ceps.iterrows():
                props = {'row_id': int(r['row_id']) if pd.notna(r['row_id']) else None, 'seq': int(r['seq']), 'status': str(r['status'])}
                if pd.notna(r['nearest_dist_m']):
                    props['nearest_dist_m'] = float(r['nearest_dist_m'])
                geom = {'type': 'Point', 'coordinates': [float(r['x']), float(r['y'])]}
                feats_out.append({'type': 'Feature', 'geometry': geom, 'properties': props})
            out_fc = {'type': 'FeatureCollection', 'features': feats_out}
            if crs:
                try:
                    out_fc['crs'] = {'type': 'name', 'properties': {'name': str(crs)}}
                except Exception:
                    pass
            os.makedirs(os.path.dirname(output_geojson), exist_ok=True)
            with open(output_geojson, 'w', encoding='utf-8') as f:
                json.dump(out_fc, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print('Erreur export GeoJSON ceps:', e)

    return df_ceps


 



if __name__ == "__main__":
    main()