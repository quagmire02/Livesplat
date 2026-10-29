"""Separate ambiguous table-contact splats without deleting captured appearance.

Contact splats and captured reflections stay at the capture location. They are
hidden once the object moves; a table patch covers the unobserved surface.
The original PLY/labels and collision geometry are left untouched.
"""
import numpy as np
from scipy.spatial import ConvexHull, QhullError

from scene_io.splat import quat_wxyz_to_mat
from segment.cluster import in_polygon


def split_contact(a, radius):
    """Boolean masks for movable body and stationary, conditional contact layer."""
    p, s = a["means"], a["scales"]
    band = 0.025 * radius
    core = p[(p[:, 2] > 0.04 * radius) & (p[:, 2] < 0.20 * radius) & (a["opacity"] > 0.15)]
    inside = np.ones(len(p), bool)
    if len(core) >= 4:
        lo, hi = np.percentile(core[:, :2], [2, 98], axis=0)
        core = core[((core[:, :2] >= lo) & (core[:, :2] <= hi)).all(1)]
        try:
            poly = core[ConvexHull(core[:, :2]).vertices, :2]
            c = poly.mean(0)
            inside = in_polygon(p[:, :2], c + 1.05 * (poly - c))
        except (QhullError, ValueError):
            pass
    R = quat_wxyz_to_mat(a["quats"])
    thin = s.argmin(1)
    normal_z = np.abs(R[np.arange(len(p)), 2, thin])
    flat_table = (normal_z > 0.75) & (s.min(1) < 0.5 * s.max(1))
    contact = (p[:, 2] < 0) | ((p[:, 2] < band) & (flat_table | ~inside))
    return ~contact, contact


def table_patch_polygon(a, reflection, radius):
    """Patch follows the actual removed scan footprint, never a generated collision mesh."""
    parts = [a["means"][(a["means"][:, 2] < 0.04 * radius) & (a["opacity"] > 0.05), :2]]
    if reflection is not None:
        parts.append(reflection["means"][(reflection["means"][:, 2] > -0.08 * radius)
                                          & (reflection["opacity"] > 0.05), :2])
    xy = np.concatenate(parts)
    if len(xy) < 4:
        return None
    lo, hi = np.percentile(xy, [1, 99], axis=0)
    xy = xy[((xy >= lo) & (xy <= hi)).all(1)]
    try:
        poly = xy[ConvexHull(xy).vertices]
    except (QhullError, ValueError):
        return None
    c = poly.mean(0)
    return c + 1.08 * (poly - c)


def at_capture(pose, base, radius, scale=1.0):
    if pose is None or abs(scale - 1.0) > 1e-4:
        return False
    p, q = pose
    # Include orientation: rotating in place also invalidates the baked contact/reflection.
    return bool(np.linalg.norm(np.asarray(p) - base) < max(0.002 * radius, 0.0005)
                and abs(q[0]) > np.cos(np.radians(2) / 2))
