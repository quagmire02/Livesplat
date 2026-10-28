"""Stage 2: objects = Gaussians above the table plane, inside the table outline, grouped by DBSCAN.

All lengths are fractions of the table radius, so they work before metric scale is fixed.
"""
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN


def in_polygon(xy, poly):
    """Even-odd ray casting, vectorised over points."""
    x, y = xy[:, 0], xy[:, 1]
    inside = np.zeros(len(xy), bool)
    for (x1, y1), (x2, y2) in zip(poly, np.roll(poly, -1, 0)):
        cross = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1)
        inside ^= cross
    return inside


def cluster_objects(pts_t, opacity, polygon, radius, min_h=0.015, max_h=1.5, eps=0.03,
                    min_opacity=0.15, min_size=0.03, voxel=0.008, min_count=200, min_frac=0.05):
    """pts_t: Gaussian centres in the table frame. Returns labels (N,) int: 0 = background."""
    r = radius
    inside = in_polygon(pts_t[:, :2], polygon * 0.98)
    cand = inside & (pts_t[:, 2] > min_h * r) & (pts_t[:, 2] < max_h * r) & (opacity > min_opacity)
    idx = np.flatnonzero(cand)
    labels = np.zeros(len(pts_t), np.int32)
    if len(idx) == 0:
        return labels
    # Voxel-downsample for DBSCAN, then give every candidate its voxel's label.
    v = np.floor(pts_t[idx] / (voxel * r)).astype(np.int64)
    uniq, inv = np.unique(v, axis=0, return_inverse=True)
    centres = (uniq + 0.5) * voxel * r
    lab = DBSCAN(eps=eps * r, min_samples=4).fit_predict(centres)
    per = lab[inv.ravel()]

    # Keep clusters that are tall/wide enough; relabel 1..K by size.
    keep = []
    for k in set(lab) - {-1}:
        P = pts_t[idx[per == k]]
        ext = np.percentile(P, 98, 0) - np.percentile(P, 2, 0)
        if max(ext) > min_size * r:
            keep.append((len(P), k))
    if keep:                                    # drop fragments: tiny next to the largest object
        big = max(n for n, _ in keep)
        keep = [(n, k) for n, k in keep if n >= max(min_count, min_frac * big)]
    for new, (_, k) in enumerate(sorted(keep, reverse=True), 1):
        labels[idx[per == k]] = new
    return labels


def grow_labels(pts_t, labels, radius, polygon, margin=0.025, min_h=0.004, iters=3):
    """Region growing: unlabelled Gaussians above the table that lie within `margin` (x table radius)
    of an object join it, any opacity. Removes the "ghost" left behind when an object moves."""
    out = labels.copy()
    cand = np.flatnonzero((out == 0) & (pts_t[:, 2] > min_h * radius)
                          & in_polygon(pts_t[:, :2], polygon * 1.05))
    for _ in range(iters):
        obj = np.flatnonzero(out > 0)
        if len(obj) == 0 or len(cand) == 0:
            break
        d, j = cKDTree(pts_t[obj]).query(pts_t[cand], distance_upper_bound=margin * radius)
        hit = np.isfinite(d)
        if not hit.any():
            break
        out[cand[hit]] = out[obj[j[hit]]]
        cand = cand[~hit]
    return out


def attach_base_and_reflection(pts_t, labels, radius, base_band=0.02, refl_depth=0.6, grow=1.15,
                               low_frac=20):
    """Inside each object's footprint (hull of its lowest points, scaled by `grow`):
    - Gaussians within +-base_band (x radius) of the plane join the object (its base; the table
      under it was never seen anyway).
    - Gaussians below that, down to refl_depth, get label -k: the object's mirror image in a glossy
      table. The viewer shows them only while object k is at rest."""
    from scipy.spatial import ConvexHull
    out = labels.copy()
    z = pts_t[:, 2] / radius
    for k in range(1, labels.max() + 1):
        Q = pts_t[labels == k]
        if len(Q) < 10:
            continue
        low = Q[Q[:, 2] <= np.percentile(Q[:, 2], low_frac)]
        hull = low[ConvexHull(low[:, :2]).vertices, :2]
        c = hull.mean(0)
        inside = in_polygon(pts_t[:, :2], c + (hull - c) * grow) & (out == 0)
        out[inside & (np.abs(z) <= base_band)] = k
        out[inside & (z < -base_band) & (z > -refl_depth)] = -k
    return out


def clean_edges(pts_t, scales, opacity, labels, max_scale_frac=0.5):
    """Drop Gaussians that are large relative to their object, or isolated from it."""
    out = labels.copy()
    for k in range(1, labels.max() + 1):
        m = np.flatnonzero(labels == k)
        P = pts_t[m]
        size = np.linalg.norm(np.percentile(P, 98, 0) - np.percentile(P, 2, 0))
        big = scales[m].max(1) > max_scale_frac * size * 0.2
        d, _ = cKDTree(P).query(P, k=6)
        lonely = d[:, -1] > 4 * np.median(d[:, -1])
        out[m[big | lonely]] = 0
    return out
