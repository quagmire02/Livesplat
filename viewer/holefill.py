"""Fill the table area under an object (never seen by the camera) once the object moves away.

A flat grid of Gaussians on the object's footprint, coloured by inverse-distance blending of the
table Gaussians in a ring around the footprint. Shown only while the object is away from rest.
"""
import numpy as np
from scipy.spatial import ConvexHull, cKDTree

from segment.cluster import in_polygon


def footprint(hull_local, grow=1.15):
    """Footprint polygon (local xy, around the base origin) from the lowest hull vertices."""
    h = hull_local
    low = h[h[:, 2] <= h[:, 2].min() + 0.15 * np.ptp(h[:, 2])]
    pts = low[:, :2] if len(low) >= 3 else h[:, :2]
    poly = pts[ConvexHull(pts).vertices]
    c = poly.mean(0)
    return c + (poly - c) * grow


def fill_splat(bg, base, hull_local, radius, cells=48, ring=(1.05, 1.8), k=8, polygon=None):
    """bg: activated background splat in the table frame. Returns an activated splat in the
    object's local frame (origin = base), or None if no table colour is found around it."""
    poly = footprint(hull_local) if polygon is None else polygon
    c = poly.mean(0)
    r_fp = np.linalg.norm(poly - c, axis=1).max()
    cell = 2 * r_fp / cells
    gx, gy = np.meshgrid(np.arange(-r_fp, r_fp, cell), np.arange(-r_fp, r_fp, cell))
    g = np.c_[gx.ravel(), gy.ravel()] + c
    g = g[in_polygon(g, poly)]
    if len(g) == 0:
        return None

    m = bg["means"]
    d = np.linalg.norm(m[:, :2] - (base[:2] + c), axis=1)
    sel = (np.abs(m[:, 2]) < 0.02 * radius) & (bg["opacity"] > 0.2) & (d > ring[0] * r_fp) & (d < ring[1] * r_fp)
    if sel.sum() < k:
        return None
    ring_xy = m[sel, :2] - base[:2]
    ring_rgb = bg["rgb"][sel]
    lum = ring_rgb @ np.array([0.299, 0.587, 0.114])           # drop outliers (e.g. a paper sheet)
    mad = np.median(np.abs(lum - np.median(lum))) + 1e-3
    ok = np.abs(lum - np.median(lum)) < 2.5 * mad
    if ok.sum() >= k:
        ring_xy, ring_rgb = ring_xy[ok], ring_rgb[ok]
    dist, idx = cKDTree(ring_xy).query(g, k=k)
    w = 1.0 / np.maximum(dist, 1e-4) ** 2
    rgb = (ring_rgb[idx] * w[..., None]).sum(1) / w.sum(1, keepdims=True)

    n = len(g)
    # Feather: opacity falls off over the last 4 cells to the footprint edge (no hard square).
    a, b = poly, np.roll(poly, -1, 0)
    ab = b - a
    t = np.clip(((g[:, None] - a[None]) * ab[None]).sum(-1) / (ab * ab).sum(-1)[None], 0, 1)
    edge = np.linalg.norm(g[:, None] - (a[None] + t[..., None] * ab[None]), axis=-1).min(1)
    alpha = 0.97 * np.clip(edge / (4 * cell), 0.15, 1.0)
    return {"means": np.c_[g, np.full(n, 0.0005)].astype(np.float32),
            "rgb": rgb.astype(np.float32),
            "opacity": alpha.astype(np.float32),
            "scales": np.tile([cell * 0.75, cell * 0.75, cell * 0.08], (n, 1)).astype(np.float32),
            "quats": np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)).astype(np.float32)}
