"""Mouse picking: which object does a camera ray hit first? Exact ray vs convex hull (slab test)."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def hull_planes(hull_pts):
    """Half-spaces n.x + d <= 0 of a convex hull, as (n (F,3), d (F,))."""
    eq = ConvexHull(hull_pts).equations
    return eq[:, :3], eq[:, 3]


def ray_hits_hull(o, v, n, d):
    """Entry distance t >= 0 of ray o + t v into the convex set, or None."""
    t0, t1 = 0.0, np.inf
    nv, no = n @ v, n @ o + d
    for a, b in zip(nv, no):
        if abs(a) < 1e-12:
            if b > 0:
                return None
            continue
        t = -b / a
        if a < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return None
    return t0


def pick(ray_o, ray_d, objects, world):
    """objects: dicts with id and hull (local frame; scale changes included). Nearest hit id or None."""
    ray_o, ray_d = np.asarray(ray_o, float), np.asarray(ray_d, float)
    ray_d = ray_d / np.linalg.norm(ray_d)
    best, best_t = None, np.inf
    for o in objects:
        k = o["id"]
        if not world.present(k):
            continue
        p, q = world.pose(k)
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        lo, lv = R.T @ (ray_o - p), R.T @ ray_d                     # ray in the object's local frame
        t = ray_hits_hull(lo, lv, *hull_planes(o["hull"]))
        if t is not None and t < best_t:
            best, best_t = k, t
    return best
