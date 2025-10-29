import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks
from skimage.transform import hough_line, hough_line_peaks


def fft2_analysis(counts, voxel_size):
    """Analyse 2D par FFT du champ de densité. Retourne (angle_deg, spacing_m, mag, (FX,FY))."""
    img = counts.astype(float)
    if img.sum() == 0:
        raise ValueError("Image vide pour FFT.")
    ny, nx = img.shape
    img_hp = img - ndimage.gaussian_filter(img, sigma=3)
    F = np.fft.fftshift(np.fft.fft2(img_hp))
    mag = np.abs(F)
    fx = np.fft.fftshift(np.fft.fftfreq(nx, d=1.0))
    fy = np.fft.fftshift(np.fft.fftfreq(ny, d=1.0))
    FX, FY = np.meshgrid(fx, fy)
    center = (ny//2, nx//2)
    mag[center] = 0
    idx_flat = np.argmax(mag)
    iy, ix = np.unravel_index(idx_flat, mag.shape)
    fpx = FX[iy, ix]
    fpy = FY[iy, ix]
    f_norm = np.hypot(fpx, fpy)
    if f_norm == 0:
        raise ValueError("Pas de pic spectral significatif trouvé.")
    angle_rad = np.arctan2(fpy, fpx) + np.pi/2
    angle_deg = (np.degrees(angle_rad)) % 180
    spacing_m = voxel_size / f_norm
    return float(angle_deg), float(spacing_m), mag, (FX, FY)


def hough_analysis(counts, x_centers, y_centers, voxel_size, threshold_rel=0.2):
    """Binarise et applique la transformée de Hough; retourne (orientation_deg, spacing, binary)."""
    img = counts.copy().astype(float)
    if img.sum() == 0:
        raise ValueError("Image vide pour Hough.")
    img_norm = img / img.max()
    binary = img_norm >= threshold_rel
    hspace, angles, dists = hough_line(binary)
    accum, angles_peaks, dists_peaks = hough_line_peaks(hspace, angles, dists, num_peaks=6)
    if len(angles_peaks) == 0:
        raise ValueError("Aucun pic Hough trouvé.")
    dominant_angle = np.median(angles_peaks)
    line_dir = np.array([np.cos(dominant_angle + np.pi/2), np.sin(dominant_angle + np.pi/2)])
    orientation_deg = (np.degrees(np.arctan2(line_dir[1], line_dir[0]))) % 180
    rows, cols = np.nonzero(binary)
    xs = x_centers[cols]
    ys = y_centers[rows]
    pts = np.column_stack([xs, ys])
    normal = np.array([-line_dir[1], line_dir[0]])
    normal = normal / np.linalg.norm(normal)
    projections = pts @ normal
    if projections.size < 10:
        spacing = np.nan
    else:
        hist_min, hist_max = projections.min(), projections.max()
        nbins = max(50, int((hist_max - hist_min) / (voxel_size/2)))
        hist, bin_edges = np.histogram(projections, bins=nbins)
        peaks, _ = find_peaks(hist, height=np.max(hist)*0.2, distance=1)
        if len(peaks) >= 2:
            peak_positions = (bin_edges[peaks] + bin_edges[peaks+1]) / 2.0
            dists_peaks = np.diff(peak_positions)
            spacing = float(np.median(dists_peaks))
        else:
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
