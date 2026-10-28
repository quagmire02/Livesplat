"""3DGS PLY load/save. Disk = raw params (log scale, logit opacity, SH). Activate in memory only."""
import numpy as np
from plyfile import PlyData, PlyElement

SH_C0 = 0.28209479177387814


def load_raw(path) -> dict:
    v = PlyData.read(str(path))["vertex"]
    names = [p.name for p in v.properties]
    g = lambda pre: np.stack([v[n] for n in sorted((n for n in names if n.startswith(pre)),
                                                    key=lambda s: int(s.rsplit("_", 1)[1]))], 1)
    return {
        "means": np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32),
        "sh0": g("f_dc_").astype(np.float32),                     # (N,3)
        "shN": g("f_rest_").astype(np.float32) if any(n.startswith("f_rest_") for n in names)
               else np.zeros((len(v), 0), np.float32),            # (N,3*K), channel-major
        "opacity": np.asarray(v["opacity"], np.float32),          # logit
        "scales": g("scale_").astype(np.float32),                 # log
        "quats": g("rot_").astype(np.float32),                    # wxyz, unnormalised
    }


def save_raw(path, s: dict):
    N = len(s["means"])
    cols = (["x", "y", "z"] + [f"f_dc_{i}" for i in range(3)]
            + [f"f_rest_{i}" for i in range(s["shN"].shape[1])] + ["opacity"]
            + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])
    data = np.concatenate([s["means"], s["sh0"], s["shN"], s["opacity"][:, None],
                           s["scales"], s["quats"]], 1).astype(np.float32)
    arr = np.empty(N, dtype=[(c, "f4") for c in cols])
    for i, c in enumerate(cols):
        arr[c] = data[:, i]
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))


def subset(s: dict, mask) -> dict:
    return {k: v[mask] for k, v in s.items()}


def activate(s: dict) -> dict:
    """Raw -> render-ready arrays (float32): rgb [0,1], opacity [0,1], scales (m), unit quats wxyz."""
    q = s["quats"] / np.linalg.norm(s["quats"], axis=1, keepdims=True)
    return {
        "means": s["means"],
        "rgb": np.clip(0.5 + SH_C0 * s["sh0"], 0, 1).astype(np.float32),
        "opacity": (1 / (1 + np.exp(-s["opacity"]))).astype(np.float32),
        "scales": np.exp(s["scales"]).astype(np.float32),
        "quats": q.astype(np.float32),
    }


def quat_wxyz_to_mat(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)


def covariances(a: dict):
    """Activated splat -> (N,3,3) covariance = R S S^T R^T."""
    R = quat_wxyz_to_mat(a["quats"])
    M = R * a["scales"][:, None, :]
    return (M @ M.transpose(0, 2, 1)).astype(np.float32)


def transform(s: dict, T: np.ndarray, scale: float = 1.0) -> dict:
    """Apply world transform x' = scale * R x + t to a raw splat (means, rotations, log scales)."""
    from scipy.spatial.transform import Rotation
    R, t = T[:3, :3], T[:3, 3]
    out = dict(s)
    out["means"] = (scale * s["means"] @ R.T + t).astype(np.float32)
    q = s["quats"][:, [1, 2, 3, 0]]                     # wxyz -> xyzw for scipy
    q2 = (Rotation.from_matrix(R) * Rotation.from_quat(q)).as_quat()
    out["quats"] = q2[:, [3, 0, 1, 2]].astype(np.float32)
    out["scales"] = (s["scales"] + np.log(scale)).astype(np.float32)
    return out
