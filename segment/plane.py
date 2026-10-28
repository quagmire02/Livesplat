"""Stage 1: find the table plane and build the table frame (Z up, origin at table centre).

Heuristic: the cameras orbit the table, so their optical axes meet near the table centre
("look-at point"). Among RANSAC planes, the table is the one with an inlier patch at the
look-at point. The normal points toward the cameras.
"""
import cv2
import numpy as np
from scipy.spatial import ConvexHull



def look_at_point(c2ws):
    """Least-squares point closest to all camera optical axes (OpenCV: +z forward)."""
    A, b = np.zeros((3, 3)), np.zeros(3)
    for T in c2ws:
        o, d = T[:3, 3], T[:3, 2] / np.linalg.norm(T[:3, 2])
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ o
    return np.linalg.solve(A, b)


def ransac_plane(P, thresh, iters=2000, rng=None):
    rng = rng or np.random.default_rng(0)
    best, best_n = None, 0
    for _ in range(iters):
        s = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-12:
            continue
        n /= np.linalg.norm(n)
        d = -n @ s[0]
        k = int((np.abs(P @ n + d) < thresh).sum())
        if k > best_n:
            best, best_n = (n, d), k
    n, d = best
    inl = np.abs(P @ n + d) < thresh
    # Refine with PCA on inliers.
    c = P[inl].mean(0)
    n = np.linalg.svd(P[inl] - c, full_matrices=False)[2][2]
    return n, -n @ c


def orbit_up(cams, look):
    """Up = normal of the camera orbit (smallest PCA axis), signed so cameras are above the look-at point."""
    c = cams - cams.mean(0)
    up = np.linalg.svd(c, full_matrices=False)[2][2]
    return up if (cams.mean(0) - look) @ up > 0 else -up


def find_table(means, opacity, c2ws, cyl_frac=0.35, band_frac=0.01, thresh_frac=0.004, min_opacity=0.5,
               fill_opacity=0.1, obj_h_frac=0.4, cell_frac=0.01, max_r_frac=1.2):
    """Returns dict with world->table 4x4 'T', table 'polygon' (K,2) in table frame, 'radius'.

    The table top is the densest height inside a vertical cylinder around the look-at axis:
    objects spread over many heights, the floor below is mostly hidden by the table.
    """
    look = look_at_point(c2ws)
    cams = np.array([T[:3, 3] for T in c2ws])
    scene = np.linalg.norm(cams - look, axis=1).mean()          # camera distance to look-at
    up = orbit_up(cams, look)
    P = means[opacity > min_opacity]
    rel = P - look
    h = rel @ up
    radial = np.linalg.norm(rel - np.outer(h, up), axis=1)
    cyl = P[radial < cyl_frac * scene]
    hc = h[radial < cyl_frac * scene]
    hc_cam = (cams - look) @ up
    cyl, hc = cyl[hc < hc_cam.min()], hc[hc < hc_cam.min()]     # below the cameras
    bw = band_frac * scene
    hist, edges = np.histogram(hc, bins=np.arange(hc.min(), hc.max() + bw, bw))
    peak = edges[np.argmax(hist)] + bw / 2
    band = cyl[np.abs(hc - peak) < 2 * bw]
    thresh = thresh_frac * scene
    n, d = ransac_plane(band, thresh)
    if n @ up < 0:
        n, d = -n, -d
    if n @ up < 0.9:
        raise RuntimeError(f"table plane tilted {np.degrees(np.arccos(n @ up)):.0f} deg from orbit up")
    inl = P[np.abs(P @ n + d) < thresh]

    # Table extent: 2D occupancy flood fill in the plane. Cells are occupied by table points
    # or by object points just above the plane (objects hide the table under them).
    # The air gap to walls and other furniture stops the fill.
    foot = look - (look @ n + d) * n
    ex = np.cross(n, [0, 0, 1.0]) if abs(n[2]) < 0.9 else np.cross(n, [1.0, 0, 0])
    ex /= np.linalg.norm(ex)
    ey = np.cross(n, ex)
    Q = means[opacity > fill_opacity]
    hq = Q @ n + d
    Q = Q[(hq > -thresh) & (hq < obj_h_frac * scene)]
    uv = np.stack([(Q - foot) @ ex, (Q - foot) @ ey], 1)
    cell = cell_frac * scene
    half = int(max_r_frac * scene / cell)
    ij = np.floor(uv / cell).astype(int) + half
    ok = (ij >= 0).all(1) & (ij < 2 * half).all(1)
    occ = np.zeros((2 * half, 2 * half), np.uint8)
    occ[ij[ok, 1], ij[ok, 0]] = 1
    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    _, comp = cv2.connectedComponents(occ, connectivity=8)
    k = comp[half, half]
    if k == 0:                                                    # foot cell empty: nearest component
        ys, xs = np.nonzero(comp)
        j = np.argmin((ys - half) ** 2 + (xs - half) ** 2)
        k = comp[ys[j], xs[j]]
    in_comp = np.zeros(len(inl), bool)
    iuv = np.floor(np.stack([(inl - foot) @ ex, (inl - foot) @ ey], 1) / cell).astype(int) + half
    okk = (iuv >= 0).all(1) & (iuv < 2 * half).all(1)
    in_comp[okk] = comp[iuv[okk, 1], iuv[okk, 0]] == k
    table = inl[in_comp]
    ys, xs = np.nonzero(comp == k)                                # footprint incl. objects
    footprint = np.stack([(xs - half + 0.5) * cell, (ys - half + 0.5) * cell], 1)

    # Table frame: z = n, x = arbitrary in-plane axis, origin = table centroid on the plane.
    x = np.cross(n, [0, 0, 1.0]) if abs(n[2]) < 0.9 else np.cross(n, [1.0, 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(n, x)
    R = np.stack([x, y, n])                                       # rows: world -> table
    o = table.mean(0)
    o = o - (o @ n + d) * n
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ o
    off = np.array([(o - foot) @ x, (o - foot) @ y])               # foot frame -> table frame
    fp = footprint - off
    poly = fp[ConvexHull(fp).vertices]
    return {"T": T, "polygon": poly, "radius": float(np.linalg.norm(poly, axis=1).max()),
            "scene_scale": float(scene), "plane_thresh": float(thresh), "n_table": int(len(table))}


def to_table(T, pts):
    return pts @ T[:3, :3].T + T[:3, 3]
