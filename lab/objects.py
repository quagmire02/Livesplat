"""Objects for lab tests: the scan's objects with the best available collision shape.

If a full 3D model exists (viewer.model3d -> gen/model_<id>/model.glb), its convex hull replaces the scan hull:
the scan never sees backs and undersides, so scan hulls are often too thin (e.g. the can: 8 cm deep), which
makes objects tip too easily. Objects are deep copies; the viewer's scene is untouched.
"""
import copy
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull


def model_hull(run, k):
    """Convex-hull vertices of the object's 3D model in its object frame (z-up, base at z=0), or None."""
    p = Path(run) / "gen" / f"model_{k}" / "model.glb"
    if not p.exists():
        return None
    import trimesh
    v = np.asarray(trimesh.load(p, force="mesh").vertices, float)
    v = np.c_[v[:, 0], -v[:, 2], v[:, 1]]                       # glb (x, z, -y) -> object frame (x, y, z)
    v[:, 2] -= v[:, 2].min()
    v[v[:, 2] < 0.04 * v[:, 2].max(), 2] = 0.0                  # it stood on the table: flat base, not a rounded one
    return v[ConvexHull(v).vertices].astype(np.float32)


def lab_objects(sc, run):
    """[(object dict, shape source)] for every object."""
    from physics.world import stands_alone
    out = []
    for o in sc.objects:
        c = copy.deepcopy({k: v for k, v in o.items() if k not in ("splat", "contact", "refl", "fill", "appearance")})
        h = model_hull(run, o["id"])
        if h is not None and stands_alone(h, c["mass"]):         # the object stood still in the scan
            c["hull"] = h
            out.append((c, "3D model"))
        else:
            out.append((c, "scan hull" if h is None else "scan hull (3D model did not stand)"))
    return out
