#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voxelisation LIDAR (vigne) + affichage
--------------------------------------
- Fonction voxelize_xy_points(df, voxel): agrège X,Y (et Z / z_norm si dispo) en voxels/cellules XY.
- Fonction show_voxel_maps(vox, ...): affiche 1) carte de densité (#points/voxel), 2) carte de hauteur (p95).
- CLI: permet d'exécuter sur un CSV (colonnes X,Y,[Z],[z_norm],[Classification]) et de sauver PNG + Parquet.

Dépendances: pandas, numpy, matplotlib
Optionnel: filtrage par classes végétation (ex: 3 4).
"""

from __future__ import annotations
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple

def voxelize_xy_points(df: pd.DataFrame, voxel: float = 0.20) -> pd.DataFrame:
    """
    Voxelise en XY et calcule des statistiques par cellule.
    Colonnes attendues: X, Y ; optionnelles: z_norm ou Z.
    Sorties:
        DataFrame avec colonnes:
            ix, iy            : indices de cellule
            x_center, y_center: centre géométrique (m)
            n                 : nombre de points
            z_p95, z_mean, z_std (si Z/z_norm présent)
            density           : n (points/cellule)
            grid_origin_x/y   : origine de la grille (pour regrillage)
            voxel             : taille de cellule
    """
    if not {"X","Y"}.issubset(df.columns):
        raise ValueError("Colonnes requises: X, Y (et optionnellement z_norm ou Z).")

    x0 = float(df["X"].min())
    y0 = float(df["Y"].min())

    ix = np.floor((df["X"].values - x0) / voxel).astype(int)
    iy = np.floor((df["Y"].values - y0) / voxel).astype(int)

    dat = pd.DataFrame({"ix": ix, "iy": iy, "X": df["X"].values, "Y": df["Y"].values})

    # Quelle colonne pour la hauteur ?
    zcol = None
    if "z_norm" in df.columns:
        zcol = "z_norm"
    elif "Z" in df.columns:
        zcol = "Z"

    agg = {
        "X": "mean",
        "Y": "mean",
        "ix": "size"
    }
    if zcol is not None:
        agg[zcol] = ["mean", "std", (lambda v: np.percentile(v, 95))]

    g = dat.join(df[[zcol]] if zcol is not None else pd.DataFrame(index=dat.index)).groupby(["ix","iy"]).agg(agg)
    # Aplatir colonnes
    cols = ["x_center","y_center","n"]
    if zcol is not None:
        cols += [f"{zcol}_mean", f"{zcol}_std", f"{zcol}_p95"]
    vox = g.reset_index()
    new_cols = ["ix","iy"] + cols
    vox.columns = new_cols
    # Densité = nombre de points par cellule
    vox["density"] = vox["n"].astype(float)
    vox["grid_origin_x"] = x0
    vox["grid_origin_y"] = y0
    vox["voxel"] = float(voxel)
    return vox

def _grid_from_vox(vox: pd.DataFrame, value_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruit une image 2D (matrice) à partir de ix,iy et d'une colonne valeur.
       Retourne (img, x_edges, y_edges) pour affichage avec extent."""
    ix = vox["ix"].to_numpy()
    iy = vox["iy"].to_numpy()
    val = vox[value_col].to_numpy().astype(float)
    nx = int(ix.max() - ix.min() + 1)
    ny = int(iy.max() - iy.min() + 1)
    # décaler pour commencer à 0
    ix0 = ix - ix.min()
    iy0 = iy - iy.min()
    img = np.full((ny, nx), np.nan, dtype=float)
    img[iy0, ix0] = val
    x0 = vox["grid_origin_x"].iloc[0] + vox["voxel"].iloc[0]*ix.min()
    y0 = vox["grid_origin_y"].iloc[0] + vox["voxel"].iloc[0]*iy.min()
    x_edges = np.arange(nx+1)*vox["voxel"].iloc[0] + x0
    y_edges = np.arange(ny+1)*vox["voxel"].iloc[0] + y0
    return img, x_edges, y_edges

def show_voxel_maps(vox: pd.DataFrame, out_prefix: Optional[str] = None):
    """
    Affiche 2 cartes: densité (#points/cellule) et hauteur p95 (si dispo).
    Sauvegarde en PNG si out_prefix est fourni.
    Rappel: 1 figure par chart, pas de style/couleur imposés (respect des contraintes).
    """
    # 1) densité
    img_d, xe, ye = _grid_from_vox(vox, "density")
    plt.figure()
    plt.imshow(img_d, origin="lower", extent=[xe[0], xe[-1], ye[0], ye[-1]], aspect="equal")
    plt.title("Densité (points par cellule)")
    plt.xlabel("X (m)"); plt.ylabel("Y (m)")
    if out_prefix:
        plt.savefig(f"{out_prefix}_density.png", dpi=200, bbox_inches="tight")
    plt.show()

 

def main(csv_path: str, voxel: float, veg: Optional[Tuple[int, ...]]):
    base, _ = os.path.splitext(csv_path)
    df = pd.read_csv(csv_path)
    # Filtrage classes végétation si demandé
    if veg and "Classification" in df.columns:
        df = df[df["Classification"].isin(veg)].copy()

    vox = voxelize_xy_points(df, voxel=voxel)
    # Export parquet
    vox.to_parquet(f"{base}_vox.parquet")
    # Affichages
    show_voxel_maps(vox, out_prefix=base)
    print("Voxelisation terminée.")
    print(f"Export: {base}_vox.parquet ; Figures (si z/z_norm) : {base}_density.png, {base}_p95.png")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Voxelisation LIDAR (vigne) + affichage densité/hauteur")
    ap.add_argument("csv", help="Chemin vers le CSV (colonnes X,Y,[Z],[z_norm],[Classification])")
    ap.add_argument("--voxel", type=float, default=0.20, help="Taille de cellule XY (m)")
    ap.add_argument("--veg", type=int, nargs="+", default=None, help="Classes végétation à garder (ex: --veg 3 4)")
    args = ap.parse_args()
    main(args.csv, args.voxel, tuple(args.veg) if args.veg else None)
