import os
import json
import csv


def export_voxels_to_csv(path, counts, x_centers, y_centers, mask=None, labels=None):
    if mask is None:
        mask = counts > 0
    rows, cols = __import__('numpy').where(mask)
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
    if mask is None:
        mask = counts > 0
    rows, cols = __import__('numpy').where(mask)
    features = []
    idx = 0
    for r, c in zip(rows, cols):
        cnt = int(counts[r, c])
        props = {'count': cnt}
        if labels is not None:
            props['cluster'] = int(labels[idx])
        if as_polygons:
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
