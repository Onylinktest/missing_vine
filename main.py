import argparse
import os
import sys
import pandas as pd
import numpy as np
from scipy import fftpack
from scipy.signal import correlate2d
from skimage.transform import hough_line, hough_line_peaks

#!/usr/bin/env python3
"""
main.py

Lecture de data/parcel.csv (fid,X,Y,Z,Intensity,Classification),
filtrage sur Classification == 4 (végétation) et affichage 2D (X,Y).

Usage:
    python main.py --input data/parcel.csv --class 4 --output output/points_class4.png
"""


import matplotlib.pyplot as plt



def load_csv(path):
        if not os.path.isfile(path):
                print(f"Fichier introuvable: {path}")
                sys.exit(1)
        df = pd.read_csv(path)
        # normaliser noms de colonnes pour tolérer différences de casse/espaces
        df.columns = df.columns.str.strip().str.lower()
        return df


def filter_class(df, class_value):
        if 'classification' not in df.columns:
                print("La colonne 'Classification' est absente du fichier CSV.")
                print("Colonnes trouvées:", ", ".join(df.columns))
                sys.exit(1)
        # convertir en numérique si besoin
        df['classification'] = pd.to_numeric(df['classification'], errors='coerce')
        if pd.isna(class_value):
                # si class_value non convertible, renvoyer vide
                return df.iloc[0:0]
        return df[df['classification'] == class_value]


def plot_xy(df, output_path=None, show=True):
        if 'x' not in df.columns or 'y' not in df.columns:
                print("Colonnes 'X' et/ou 'Y' absentes du fichier CSV.")
                print("Colonnes trouvées:", ", ".join(df.columns))
                sys.exit(1)

        x = df['x']
        y = df['y']

        plt.figure(figsize=(8, 8))
        plt.scatter(x, y, s=1, c='green', alpha=0.7, linewidths=0)
        plt.title("Points LiDAR - Classification = 4 (végétation)")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True)

        if output_path:
                os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"Image sauvegardée: {output_path}")

        if show:
                plt.show()
        else:
                plt.close()


def plot_xyz(df, output_path=None, show=True, elev=20, azim=-60):
    # affichage 3D des points (X,Y,Z)
    from mpl_toolkits.mplot3d import Axes3D  # nécessaire pour certains backends matplotlib
    if 'x' not in df.columns or 'y' not in df.columns or 'z' not in df.columns:
        print("Colonnes 'X', 'Y' et/ou 'Z' absentes du fichier CSV.")
        print("Colonnes trouvées:", ", ".join(df.columns))
        sys.exit(1)

    x = df['x']
    y = df['y']
    z = df['z']

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=1, c='green', alpha=0.7, linewidths=0)
    ax.set_title("Points LiDAR - Classification (3D)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Image 3D sauvegardée: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


# Trois méthodes pour estimer l'orientation dominante et l'espacement inter-rangs
# à partir d'un nuage de points (X, Y). Ces fonctions sont au niveau module (pas imbriquées).
# Dépendances : numpy, scipy, matplotlib, skimage

import matplotlib.pyplot as plt


def _extract_xy(points):
    """Accepte DataFrame, (N,2) numpy array ou tuple/list (x,y). Renvoie x,y numpy 1D."""
    try:
        if isinstance(points, pd.DataFrame):
            cols = {c.lower(): c for c in points.columns}
            if 'x' in cols and 'y' in cols:
                return points[cols['x']].to_numpy(), points[cols['y']].to_numpy()
    except Exception:
        pass

    arr = np.asarray(points)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr[:, 0].astype(float), arr[:, 1].astype(float)
    if isinstance(points, (tuple, list)) and len(points) == 2:
        x = np.asarray(points[0]).astype(float)
        y = np.asarray(points[1]).astype(float)
        if x.shape == y.shape:
            return x, y
    raise ValueError("Entrée non reconnue. Attendu DataFrame (X,Y), array (N,2) ou (x,y).")


def _compute_hist2d(x, y, resolution):
    if len(x) == 0:
        raise ValueError("Aucun point fourni.")
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    pad = 1e-9
    xrange = xmax - xmin if xmax > xmin else 1.0
    yrange = ymax - ymin if ymax > ymin else 1.0
    xmin -= pad * xrange
    xmax += pad * xrange
    ymin -= pad * yrange
    ymax += pad * yrange

    nx = max(8, int(np.ceil((xmax - xmin) / resolution)))
    ny = max(8, int(np.ceil((ymax - ymin) / resolution)))
    H, xedges, yedges = np.histogram2d(x, y, bins=[nx, ny], range=[[xmin, xmax], [ymin, ymax]])
    H = H.T
    return H, xedges, yedges, resolution


def _angle_from_frequency(fx, fy):
    ang = np.degrees(np.arctan2(fy, fx))
    ang = ((ang + 90) % 180) - 90
    return ang


def detect_vine_fft(points, resolution=0.2, mask_radius_px=3, plot=False):
    x, y = _extract_xy(points)
    try:
        H, xedges, yedges, res = _compute_hist2d(x, y, resolution)
    except ValueError:
        return np.nan, np.nan

    ny, nx = H.shape
    F = np.fft.fftshift(np.fft.fft2(H))
    mag = np.abs(F)
    dx = res
    dy = res
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dy)
    FX, FY = np.meshgrid(fx, fy)

    cy, cx = ny // 2, nx // 2
    r = int(max(1, mask_radius_px))
    yy, xx = np.ogrid[:ny, :nx]
    mask_center = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    mag_masked = mag.copy()
    mag_masked[mask_center] = 0.0

    if np.all(mag_masked == 0):
        return np.nan, np.nan

    iy, ix = np.unravel_index(np.argmax(mag_masked), mag_masked.shape)
    fx0 = FX[iy, ix]
    fy0 = FY[iy, ix]
    freq_norm = np.hypot(fx0, fy0)
    if freq_norm <= 0:
        spacing = np.nan
    else:
        spacing = 1.0 / freq_norm

    angle_deg = _angle_from_frequency(fx0, fy0)

    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs[0].scatter(x, y, s=1)
        axs[0].set_title("Points (aperçu)")
        im = axs[1].imshow(np.log1p(mag), origin='lower', extent=[fx.min(), fx.max(), fy.min(), fy.max()], aspect='auto')
        axs[1].scatter([fx0], [fy0], color='red', marker='x')
        axs[1].set_xlabel('fx (1/m)')
        axs[1].set_ylabel('fy (1/m)')
        axs[1].set_title(f"FFT magnitude (log). Peak -> angle={angle_deg:.1f}°, spacing={spacing:.2f} m")
        fig.colorbar(im, ax=axs[1], label='log(magnitude)')
        plt.tight_layout()
        plt.show()

    return float(angle_deg), float(spacing)


def detect_vine_hough(points, resolution=0.2, percentile=90, plot=False, min_distance_lines=2):
    x, y = _extract_xy(points)
    try:
        H, xedges, yedges, res = _compute_hist2d(x, y, resolution)
    except ValueError:
        return np.nan, np.nan

    nonzero = H[H > 0]
    if nonzero.size == 0:
        return np.nan, np.nan
    thresh = np.percentile(nonzero, percentile)
    binary = H > thresh

    hspace, angles, dists = hough_line(binary)
    accums, thetas, rhos = hough_line_peaks(hspace, angles, dists, num_peaks=20)

    if len(thetas) == 0:
        return np.nan, np.nan

    weights = accums.astype(float)
    thetas_deg = np.degrees(thetas)
    line_angles_deg = ((thetas_deg - 90.0 + 180) % 180) - 90.0
    try:
        angle_deg = float(np.arctan2(np.sum(weights * np.sin(np.radians(line_angles_deg))),
                                     np.sum(weights * np.cos(np.radians(line_angles_deg)))))
        angle_deg = np.degrees(angle_deg)
        angle_deg = ((angle_deg + 90) % 180) - 90
    except Exception:
        angle_deg = float(np.median(line_angles_deg))

    tolerance_deg = 5.0
    mask = np.abs(((line_angles_deg - angle_deg + 180) % 180) - 180) < tolerance_deg
    rhos_selected = np.array(rhos)[mask]
    acc_selected = np.array(accums)[mask]
    if rhos_selected.size < 2:
        rhos_selected = np.array(rhos)

    if rhos_selected.size >= 2:
        rhos_sorted = np.sort(rhos_selected)
        diffs = np.diff(rhos_sorted)
        diffs_nonzero = diffs[diffs > 1e-6]
        if diffs_nonzero.size == 0:
            spacing = np.nan
        else:
            spacing = float(np.median(diffs_nonzero) * resolution)
    else:
        spacing = np.nan

    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs[0].imshow(H, origin='lower', cmap='gray', interpolation='nearest')
        axs[0].set_title('Carte densité (H)')
        axs[1].imshow(binary, origin='lower', cmap='gray')
        axs[1].set_title(f'Binary (>{percentile}th pct). Detected lines:')
        ny, nx = binary.shape
        for a, t, acc in zip(rhos, thetas, accums):
            cos_t = np.cos(t)
            sin_t = np.sin(t)
            x0 = 0
            if abs(sin_t) > 1e-6:
                y0 = (a - x0 * cos_t) / sin_t
            else:
                y0 = 0
            x1 = nx
            if abs(sin_t) > 1e-6:
                y1 = (a - x1 * cos_t) / sin_t
            else:
                y1 = ny
            axs[1].plot([x0, x1], [y0, y1], '-r', linewidth=max(0.5, min(3.0, acc / 10.0)), alpha=0.8)
        axs[1].set_xlim(0, nx)
        axs[1].set_ylim(0, ny)
        axs[1].set_xlabel('x pixels')
        axs[1].set_ylabel('y pixels')
        plt.suptitle(f"Hough -> angle={angle_deg:.1f}°, spacing≈{(spacing if spacing is not None else np.nan):.2f} m")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    return float(angle_deg), float(spacing) if spacing is not None else np.nan



def detect_vine_pca(points, plot=False):
    """Estime l'orientation dominante d'un nuage (X,Y) via PCA.

    Retourne (angle_deg, spacing_m, anisotropy_r) où spacing_m = np.nan (PCA n'est pas utilisée
    pour estimer l'espacement ici) et anisotropy_r = lambda1 / lambda2 (>=1 si définie).
    Si plot=True, affiche les points et l'axe principal.
    """
    try:
        x, y = _extract_xy(points)
    except Exception:
        return np.nan, np.nan, np.nan

    if x.size < 2:
        return np.nan, np.nan, np.nan

    XY = np.vstack((x.astype(float), y.astype(float))).T
    # centrer
    mean_xy = XY.mean(axis=0)
    XYc = XY - mean_xy

    # covariance et décomposition en valeurs propres
    try:
        cov = np.cov(XYc, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
    except Exception:
        return np.nan, np.nan, np.nan

    # trier valeurs propres décroissantes
    order = np.argsort(eigvals)[::-1]
    eigvals_sorted = eigvals[order]
    eigvecs_sorted = eigvecs[:, order]

    lambda1 = float(eigvals_sorted[0])
    lambda2 = float(eigvals_sorted[1]) if eigvals_sorted.size > 1 else 0.0

    # vecteur principal = eigenvector associé à la plus grande valeur propre
    principal = eigvecs_sorted[:, 0]

    # angle en degrés par rapport à l'axe X
    angle_deg = np.degrees(np.arctan2(principal[1], principal[0]))
    angle_deg = ((angle_deg + 90) % 180) - 90

    # anisotropie: rapport lambda1 / lambda2 (si lambda2 > 0)
    if lambda2 > 0:
        anisotropy_r = lambda1 / lambda2
    else:
        anisotropy_r = np.nan

    spacing = np.nan

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.scatter(XY[:, 0], XY[:, 1], s=1)
        cx, cy = mean_xy[0], mean_xy[1]
        # dessiner l'axe principal centré (longueur arbitraire pour visibilité)
        ax.quiver(cx, cy, principal[0], principal[1], angles='xy', scale_units='xy', scale=1, color='red', width=0.003)
        ax.set_title(f'PCA orientation: {angle_deg:.1f}° (anisotropy={anisotropy_r:.2f})')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal', adjustable='box')
        plt.show()

    return float(angle_deg), float(spacing), float(anisotropy_r) if not pd.isna(anisotropy_r) else np.nan


# Exemple d'utilisation (supposant un DataFrame pandas 'df' avec colonnes X et Y):
# angle_fft, spacing_fft = detect_vine_fft(df, resolution=0.2, plot=True)
# angle_ac, spacing_ac = detect_vine_autocorr(df, resolution=0.2, plot=True)
# angle_hough, spacing_hough = detect_vine_hough(df, resolution=0.2, percentile=90, plot=True)
# print("FFT:", angle_fft, "deg,", spacing_fft, "m")

def main():
        parser = argparse.ArgumentParser(description="Afficher points 2D pour Classification=4")
        parser.add_argument("--input", "-i", default="data/parcel.csv", help="Chemin vers le CSV")
        parser.add_argument("--class", "-c", dest="class_value", type=int, default=4, help="Valeur de classification à filtrer (défaut 4)")
        parser.add_argument("--output", "-o", default="output/points_class4_2D.png", help="Chemin de sortie pour l'image (optionnel)")
        parser.add_argument("--no-show", action="store_true", help="Ne pas afficher la fenêtre matplotlib (sauvegarder seulement)")
        parser.add_argument("--3d", dest="plot_3d", action="store_true", help="Afficher également en 3D (X,Y,Z)")
        parser.add_argument("--detect", action="store_true", help="Exécuter détection d'orientation/espacement (FFT, autocorr, Hough, PCA)")
        parser.add_argument("--detect-plot", action="store_true", help="Afficher graphiques des méthodes de détection")
        parser.add_argument("--config", "-f", dest="config", help="Chemin vers fichier JSON avec arguments (optionnel)")
        args = parser.parse_args()

        # Charger options depuis fichier JSON si fourni (les clés du JSON remplacent les valeurs par défaut)
        if args.config:
                try:
                        with open(args.config, 'r', encoding='utf-8') as fh:
                                cfg = __import__('json').load(fh)
                except Exception as e:
                        print(f"Impossible de lire le fichier de configuration '{args.config}': {e}")
                        sys.exit(1)
                # mapping des clés possibles vers attributs argparse
                key_map = {
                        'input': 'input',
                        'class': 'class_value',
                        'class_value': 'class_value',
                        'output': 'output',
                        'no_show': 'no_show',
                        'no-show': 'no_show',
                        'plot_3d': 'plot_3d',
                        '3d': 'plot_3d',
                        'detect': 'detect',
                        'detect_plot': 'detect_plot'
                }
                for k, v in cfg.items():
                        attr = key_map.get(k, None)
                        if attr is None:
                                # ignorer clés inconnues
                                continue
                        # convertir types simples si nécessaire
                        if attr == 'class_value':
                                try:
                                        setattr(args, attr, int(v))
                                except Exception:
                                        setattr(args, attr, v)
                        elif attr in ('no_show', 'plot_3d', 'detect', 'detect_plot'):
                                # accepter bool ou valeurs interprétables
                                if isinstance(v, bool):
                                        setattr(args, attr, v)
                                else:
                                        val = str(v).lower()
                                        setattr(args, attr, val in ('1', 'true', 'yes', 'y'))
                        else:
                                setattr(args, attr, v)

        df = load_csv(args.input)
        total = len(df)
        df_filtered = filter_class(df, args.class_value)
        filtered_count = len(df_filtered)

        print(f"Points lus: {total}")
        print(f"Points avec Classification == {args.class_value}: {filtered_count}")

        if filtered_count == 0:
                print("Aucun point à afficher après filtrage.")
                sys.exit(1)

        # Méthodes de détection disponibles
        detection_methods = [
                ("FFT", detect_vine_fft),
                ("Hough", detect_vine_hough),
                ("PCA", detect_vine_pca),
        ]

        # Si demandé, exécuter les méthodes de détection (avant affichage pour éviter blocage par plt.show())
        if args.detect:
                results = {}
                for name, func in detection_methods:
                        try:
                                if name == 'Hough':
                                        res = func(df_filtered, resolution=0.2, percentile=85, plot=args.detect_plot)
                                elif name in ('FFT', 'Autocorr'):
                                        res = func(df_filtered, resolution=0.3, plot=args.detect_plot)
                                else:  # PCA
                                        res = func(df_filtered, plot=args.detect_plot)
                        except Exception:
                                res = (np.nan, np.nan, np.nan)

                        # normaliser retour à 3 éléments (angle, spacing, anisotropy)
                        if isinstance(res, tuple) and len(res) == 3:
                                angle, spacing, anis = res
                        elif isinstance(res, tuple) and len(res) == 2:
                                angle, spacing = res
                                anis = np.nan
                        else:
                                # valeur unique ou autre -> forcer NaN
                                angle, spacing, anis = np.nan, np.nan, np.nan

                        results[name] = (angle, spacing, anis)

                # Affichage synthétique des résultats
                print("\nRésultats détection:")
                print(f"{'Method':8s}  {'Angle(deg)':>10s}  {'Spacing(m)':>10s}  {'Anisotropy':>10s}")
                for name in [m[0] for m in detection_methods]:
                        ang, sp, an = results.get(name, (np.nan, np.nan, np.nan))
                        ang_s = f"{ang:.2f}" if not pd.isna(ang) else 'nan'
                        sp_s = f"{sp:.2f}" if not pd.isna(sp) else 'nan'
                        an_s = f"{an:.2f}" if not pd.isna(an) else 'nan'
                        print(f"{name:8s}  {ang_s:>10s}  {sp_s:>10s}  {an_s:>10s}")

        # Affichage 2D
        plot_xy(df_filtered, output_path=args.output, show=not args.no_show)

        # Si demandé, affichage 3D
        if args.plot_3d:
                plot_xyz(df_filtered, output_path=None, show=not args.no_show)

if __name__ == "__main__":
        main()