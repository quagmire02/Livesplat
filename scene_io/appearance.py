"""Evaluate the existing 3DGS spherical harmonics for Viser's RGB-only splat interface.

Directions must be camera-to-point vectors in the ORIGINAL reconstruction frame.
No super-resolution or generated texture is involved. Supports SH degrees 0..3.
"""
import numpy as np


def view_rgb(sh0, shN, directions):
    d = np.asarray(directions, np.float32)
    d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    x, y, z = d.T
    n = shN.shape[1] // 3
    if n not in (0, 3, 8, 15):
        raise ValueError(f"Unsupported SH coefficient count: {n + 1}")
    rgb = 0.5 + 0.28209479177387814 * sh0
    rest = shN.reshape(-1, 3, n) if n else None  # PLY stores channel-major coefficients
    basis = []
    if n >= 3:
        basis += [-0.4886025119029199 * y, 0.4886025119029199 * z, -0.4886025119029199 * x]
    if n >= 8:
        basis += [1.0925484305920792 * x * y, -1.0925484305920792 * y * z,
                  0.31539156525252005 * (3 * z * z - 1), -1.0925484305920792 * x * z,
                  0.5462742152960396 * (x * x - y * y)]
    if n >= 15:
        basis += [-0.5900435899266435 * y * (3 * x * x - y * y),
                  2.890611442640554 * x * y * z,
                  -0.4570457994644658 * y * (5 * z * z - 1),
                  0.3731763325901154 * z * (5 * z * z - 3),
                  -0.4570457994644658 * x * (5 * z * z - 1),
                  1.445305721320277 * z * (x * x - y * y),
                  -0.5900435899266435 * x * (x * x - 3 * y * y)]
    for j, b in enumerate(basis):
        rgb += rest[:, :, j] * b[:, None]
    return np.clip(rgb, 0, 1).astype(np.float32)
