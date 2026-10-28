"""Photoreal render of a LiveSplat sim package: Gaussian splats posed by MuJoCo bodies.

Self-contained (numpy, torch, gsplat, mujoco, plyfile); sim.export copies it into every package.

    from render import SplatRenderer
    r = SplatRenderer("path/to/sim")            # loads metadata.json, splats/, scene.xml
    img = r.render(data, "cam_f00001")          # data = mujoco.MjData; returns HxWx3 uint8

CLI: python render.py <sim_dir> [--camera cam_x] [--settle 1.0] [--out frame.png]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

SH_C0 = 0.28209479177387814


def load_ply(path):
    from plyfile import PlyData
    v = PlyData.read(str(path))["vertex"]
    names = [p.name for p in v.properties]

    def g(pre):
        cols = sorted((n for n in names if n.startswith(pre)), key=lambda s: int(s.rsplit("_", 1)[1]))
        return np.stack([v[c] for c in cols], 1).astype(np.float32) if cols else np.zeros((len(v), 0), np.float32)
    rest = g("f_rest_")
    K = rest.shape[1] // 3
    return {"means": np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32),
            "quats": g("rot_"), "scales": g("scale_"), "opac": np.asarray(v["opacity"], np.float32),
            "sh": np.concatenate([g("f_dc_")[:, None, :], rest.reshape(len(v), 3, K).transpose(0, 2, 1)], 1)}


def quat_mul(a, b):
    """Hamilton product, wxyz, batched (torch)."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def quat_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class SplatRenderer:
    def __init__(self, sim_dir, device="cuda"):
        self.dir = Path(sim_dir)
        self.dev = device
        self.meta = json.loads((self.dir / "metadata.json").read_text())
        self.cams = {c["name"]: c for c in self.meta["cameras"]}
        t = lambda a: torch.tensor(a, device=device)
        bg = load_ply(self.dir / self.meta["splats"]["background"])
        self.static = {k: t(v) for k, v in bg.items()}
        self.objs = []
        for o in self.meta["objects"]:
            entry = {"body": o["body"], "rest": np.array(o["pose"]["pos"]),
                     "capture": np.array(o.get("capture_pos", o["pose"]["pos"])), "parts": {}, "fill": None}
            base = o["files"]["splat"]
            for part, f in (("obj", base), ("refl", base.replace(".ply", "_reflection.ply"))):
                if (self.dir / f).exists():
                    s = load_ply(self.dir / f)
                    if o.get("tint_rgb") is not None:            # colour variant: scale the RGB
                        tint = np.array(o["tint_rgb"], np.float32)
                        rgb0 = 0.5 + SH_C0 * s["sh"][:, 0]
                        s["sh"][:, 0] = (rgb0 * tint - 0.5) / SH_C0
                        s["sh"][:, 1:] *= tint
                    entry["parts"][part] = {k: t(v) for k, v in s.items()}
            ff = base.replace(".ply", "_fill.ply")
            if (self.dir / ff).exists():
                entry["fill"] = {k: t(v) for k, v in load_ply(self.dir / ff).items()}
            self.objs.append(entry)
        self.sh_degree = int(round(np.sqrt(self.static["sh"].shape[1]))) - 1

    def _posed(self, data):
        parts = [self.static]
        for o in self.objs:
            b = data.body(o["body"])
            p, q = b.xpos.copy(), b.xquat.copy()
            upright = abs(p[2] - o["rest"][2]) < 0.01 and abs(q[0]) > 0.996
            if o["fill"] is not None and np.linalg.norm(p[:2] - o["capture"][:2]) > 0.01:
                parts.append(o["fill"])                          # object left its captured spot
            for name, s in o["parts"].items():
                if name == "refl":
                    if not upright:
                        continue
                    n = max(np.hypot(q[0], q[3]), 1e-9)
                    qq, pp = np.array([q[0] / n, 0, 0, q[3] / n]), np.array([p[0], p[1], 0.0])
                else:
                    qq, pp = q, p
                R = torch.tensor(quat_mat(qq), device=self.dev, dtype=torch.float32)
                parts.append({"means": s["means"] @ R.T + torch.tensor(pp, device=self.dev, dtype=torch.float32),
                              "quats": quat_mul(torch.tensor(qq, device=self.dev, dtype=torch.float32).expand_as(s["quats"]),
                                                s["quats"]),
                              "scales": s["scales"], "opac": s["opac"], "sh": s["sh"]})
        return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}

    @torch.no_grad()
    def render(self, data, camera, width=None):
        from gsplat import rasterization
        c = self.cams[camera]
        W, H = c["width"], c["height"]
        K = np.array(c["K"], np.float64)
        if width and width != W:
            s = width / W
            K[:2] *= s
            W, H = width, int(round(H * s))
        g = self._posed(data)
        vm = torch.tensor(np.linalg.inv(np.array(c["c2w_opencv"])), device=self.dev, dtype=torch.float32)
        out, _, _ = rasterization(
            means=g["means"], quats=g["quats"] / g["quats"].norm(dim=-1, keepdim=True),
            scales=torch.exp(g["scales"]), opacities=torch.sigmoid(g["opac"]), colors=g["sh"],
            viewmats=vm[None], Ks=torch.tensor(K, device=self.dev, dtype=torch.float32)[None],
            width=W, height=H, sh_degree=self.sh_degree)
        return (out[0].clamp(0, 1) * 255).byte().cpu().numpy()


def main():
    import mujoco
    from PIL import Image
    ap = argparse.ArgumentParser()
    ap.add_argument("sim_dir")
    ap.add_argument("--camera")
    ap.add_argument("--settle", type=float, default=0.0, help="seconds of physics before rendering")
    ap.add_argument("--out", default="frame.png")
    a = ap.parse_args()
    r = SplatRenderer(a.sim_dir)
    m = mujoco.MjModel.from_xml_path(str(Path(a.sim_dir) / "scene.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    for _ in range(int(a.settle / m.opt.timestep)):
        mujoco.mj_step(m, d)
    Image.fromarray(r.render(d, a.camera or next(iter(r.cams)))).save(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
