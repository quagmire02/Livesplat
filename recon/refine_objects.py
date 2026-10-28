"""Object-focused refinement ("zoom"): fine-tune and densify ONLY object Gaussians on crops around
each object at native resolution. Background Gaussians are frozen. Every 8th view is held out
and scored (PSNR on object crops) before and after, so the gain is measured, not assumed.

Runs in the ThriftSplat venv (imports thriftsplat, gsplat):
    $THRIFT_PY recon/refine_objects.py <run_dir> --src thrift_sfm --out refined --iters 3000

Reads  <run_dir>/<src>/{splat.ply, scene/, splat/pose_delta.npy}, <run_dir>/labels_geo.npy
Writes <run_dir>/<out>/{splat.ply, labels.npy, refine.json, compare_*.jpg}, export -> <src>/export
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def load_params(path, dev):
    from plyfile import PlyData
    v = PlyData.read(path)["vertex"]
    names = [p.name for p in v.properties]

    def g(pre):
        cols = sorted((n for n in names if n.startswith(pre)), key=lambda s: int(s.rsplit("_", 1)[1]))
        return np.stack([v[c] for c in cols], 1).astype(np.float32)
    N = len(v)
    rest = g("f_rest_")                                            # channel-major (3*K)
    K = rest.shape[1] // 3
    shN = rest.reshape(N, 3, K).transpose(0, 2, 1)                 # -> (N,K,3)
    p = {"means": np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32),
         "scales": g("scale_"), "quats": g("rot_"), "opacities": np.asarray(v["opacity"], np.float32),
         "sh0": g("f_dc_")[:, None, :], "shN": np.ascontiguousarray(shN)}
    return torch.nn.ParameterDict({k: torch.nn.Parameter(torch.tensor(a, device=dev)) for k, a in p.items()})


def load_frame(path, width):
    """One frame through MapAnything's own preprocessing (same crop the intrinsics refer to),
    as uint8 (H,W,3). Mirrors thriftsplat.train_gsplat.load_views, one image at a time."""
    from mapanything.utils.image import load_images
    from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT
    height = int(round(width * 294.0 / 518.0)) // 2 * 2
    v = load_images([path], resize_mode="fixed_size", size=(width, height))[0]
    nrm = IMAGE_NORMALIZATION_DICT["dinov2"]
    img = (v["img"] * nrm.std.view(1, 3, 1, 1) + nrm.mean.view(1, 3, 1, 1)).clamp(0, 1)[0]
    return (img.permute(1, 2, 0) * 255).to(torch.uint8).numpy()


def crop_box(pts_w, viewmat, K, W, H, pad=0.15, min_side=96):
    """2D bbox (x0,y0,x1,y1) of world points in a view, padded; None if not visible."""
    pc = pts_w @ viewmat[:3, :3].T + viewmat[:3, 3]
    pc = pc[pc[:, 2] > 1e-3]
    if len(pc) < 50:
        return None
    uv = pc @ K.T
    uv = uv[:, :2] / uv[:, 2:3]
    x0, y0 = np.percentile(uv, 1, 0)
    x1, y1 = np.percentile(uv, 99, 0)
    if x1 < 0 or y1 < 0 or x0 > W or y0 > H:
        return None
    w, h = max(x1 - x0, min_side), max(y1 - y0, min_side)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = w * (1 + 2 * pad), h * (1 + 2 * pad)
    x0, x1 = int(max(0, cx - w / 2)), int(min(W, cx + w / 2))
    y0, y1 = int(max(0, cy - h / 2)), int(min(H, cy + h / 2))
    if (x1 - x0) < 32 or (y1 - y0) < 32:
        return None
    return x0, y0, x1, y1


def render_crop(params, vm, K, box, sh_degree):
    from gsplat import rasterization
    x0, y0, x1, y1 = box
    Kc = K.clone()
    Kc[0, 2] -= x0
    Kc[1, 2] -= y0
    colors = torch.cat([params["sh0"], params["shN"]], 1)
    out, _, info = rasterization(
        means=params["means"], quats=params["quats"], scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]), colors=colors, viewmats=vm[None], Ks=Kc[None],
        width=x1 - x0, height=y1 - y0, sh_degree=sh_degree, packed=True)
    return out[0].permute(2, 0, 1).clamp(0, 1), info


def psnr(a, b):
    return float(-10 * torch.log10(F.mse_loss(a, b) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--src", default="thrift_sfm")
    ap.add_argument("--out", default="refined")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--width", type=int, default=0, help="crop source width; 0 = native (largest multiple of 518)")
    ap.add_argument("--grow-grad", type=float, default=3e-5, help="lower = more densification")
    ap.add_argument("--max-growth", type=float, default=3.0, help="stop densifying at this x object count")
    ap.add_argument("--means-lr", type=float, default=1.6e-5, help="fine-tune: near the end LR of the main training")
    a = ap.parse_args()

    from gsplat.strategy import DefaultStrategy
    from thriftsplat.train_gsplat import load_views, pose_delta_to_mat, save_splat, ssim

    dev = "cuda"
    src = os.path.join(a.run_dir, a.src)
    out = os.path.join(a.run_dir, a.out)
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(0)

    # Poses and the base lens from a cheap 518 px load; frames are then read one at a time at the
    # target width, and only the object crops are kept (a 4K clip would not fit in host RAM).
    scene_dir = os.path.join(src, "scene")
    _, viewmats, Ks518, W518, H518, names, _ = load_views(scene_dir, 518, "cpu")
    scene = json.load(open(os.path.join(scene_dir, "scene.json")))
    paths = [scene["frames"][n]["source"] for n in names]
    if a.width <= 0:                                              # native: largest multiple of 518 that fits
        nw = Image.open(paths[0]).size[0]
        a.width = max(518, (nw // 518) * 518)
    probe = load_frame(paths[0], a.width)
    H, W = probe.shape[:2]
    Ks = Ks518.clone()
    Ks[:, 0, :] *= W / W518                                       # per-axis, as load_views does
    Ks[:, 1, :] *= H / H518
    pd_path = os.path.join(src, "splat", "pose_delta.npy")
    if os.path.exists(pd_path):
        viewmats = pose_delta_to_mat(torch.tensor(np.load(pd_path))) @ viewmats
    viewmats, Ks = viewmats.to(dev), Ks.to(dev)
    vm_np, K_np = viewmats.cpu().numpy(), Ks.cpu().numpy()

    params = load_params(os.path.join(src, "splat.ply"), dev)
    labels = torch.tensor(np.load(os.path.join(a.run_dir, "labels_geo.npy")), device=dev)
    assert len(labels) == len(params["means"]), "labels do not match splat (re-run segment.run)"
    sh_degree = int(round(np.sqrt(params["shN"].shape[1] + 1))) - 1
    n_obj0 = int((labels > 0).sum())
    obj_ids = [int(k) for k in torch.unique(labels).tolist() if k > 0]
    print(f"{len(params['means']):,} Gaussians, {n_obj0:,} on {len(obj_ids)} objects; "
          f"{len(names)} views at {W}x{H}; sh {sh_degree}")

    # Crop boxes per (view, object), from the object's points.
    means_np = params["means"].detach().cpu().numpy()
    lab_np = labels.cpu().numpy()
    boxes = []
    for i in range(len(names)):
        for k in obj_ids:
            b = crop_box(means_np[lab_np == k], vm_np[i], K_np[i], W, H)
            if b:
                boxes.append((i, k, b))
    test = [bx for bx in boxes if bx[0] % 8 == 0]
    train = [bx for bx in boxes if bx[0] % 8 != 0]
    crops = {}
    for i in sorted({bx[0] for bx in boxes}):
        img = load_frame(paths[i], a.width)
        for j, k, b in boxes:
            if j == i:
                crops[(i, k)] = torch.from_numpy(np.ascontiguousarray(img[b[1]:b[3], b[0]:b[2]]))
    mb = sum(c.numel() for c in crops.values()) / 1e6
    print(f"{len(train)} train crops, {len(test)} held-out crops ({mb:.0f} MB in RAM)")

    def evaluate(tag):
        vals, per = [], {k: [] for k in obj_ids}
        with torch.no_grad():
            for i, k, b in test:
                pr, _ = render_crop(params, viewmats[i], Ks[i], b, sh_degree)
                gt = crops[(i, k)].to(dev).permute(2, 0, 1).float() / 255
                v = psnr(pr, gt)
                vals.append(v)
                per[k].append(v)
        res = {"mean": float(np.mean(vals)), **{f"obj{k}": float(np.mean(v)) for k, v in per.items() if v}}
        print(f"[{tag}] held-out object-crop PSNR {res['mean']:.2f} dB  "
              + "  ".join(f"obj{k} {np.mean(v):.2f}" for k, v in per.items() if v))
        return res

    def snapshot(tag, n=3):
        with torch.no_grad():
            for j, (i, k, b) in enumerate(test[:: max(1, len(test) // n)][:n]):
                pr, _ = render_crop(params, viewmats[i], Ks[i], b, sh_degree)
                gt = crops[(i, k)].permute(2, 0, 1).float() / 255
                side = torch.cat([pr.cpu(), gt], 2).permute(1, 2, 0).numpy()
                Image.fromarray((side * 255).astype(np.uint8)).save(os.path.join(out, f"compare_{tag}_{j}.jpg"))

    before = evaluate("before")
    snapshot("before")

    lrs = {"means": a.means_lr, "scales": 2.5e-3, "quats": 5e-4, "opacities": 1e-2, "sh0": 1e-3, "shN": 1e-3 / 20}
    opts = {k: torch.optim.Adam([{"params": [params[k]], "lr": v, "name": k}], eps=1e-15) for k, v in lrs.items()}
    sched = torch.optim.lr_scheduler.ExponentialLR(opts["means"], gamma=0.1 ** (1.0 / a.iters))
    strat = DefaultStrategy(refine_start_iter=200, refine_stop_iter=int(a.iters * 0.8), refine_every=100,
                            reset_every=10 ** 9, grow_grad2d=a.grow_grad, verbose=False,
                            prune_opa=-1.0)              # no pruning: it would also delete frozen background splats
    state = strat.initialize_state(scene_scale=1.0)
    state["label"] = labels                                       # kept in sync by gsplat ops

    rng = np.random.default_rng(0)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for step in range(a.iters):
        i, k, b = train[rng.integers(len(train))]
        pr, info = render_crop(params, viewmats[i], Ks[i], b, sh_degree)
        strat.step_pre_backward(params=params, optimizers=opts, state=state, step=step, info=info)
        gt = crops[(i, k)].to(dev).permute(2, 0, 1).float() / 255
        loss = 0.8 * F.l1_loss(pr, gt) + 0.2 * (1 - ssim(pr[None], gt[None]))
        loss.backward()
        frozen = state["label"] <= 0                              # background and reflections stay fixed
        for p in params.values():
            if p.grad is not None:
                p.grad[frozen] = 0
        g2d = info["means2d"].grad
        if g2d is not None:                                       # no densification of frozen splats
            g2d[frozen[info["gaussian_ids"]]] = 0
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        strat.step_post_backward(params=params, optimizers=opts, state=state, step=step, info=info, packed=True)
        sched.step()
        n_obj = int((state["label"] > 0).sum())
        if n_obj > a.max_growth * n_obj0:
            strat.refine_stop_iter = min(strat.refine_stop_iter, step)
        if step % 250 == 0 or step == a.iters - 1:
            print(f"  {step:5d}/{a.iters} loss {loss.item():.4f} object splats {n_obj:,} "
                  f"total {len(params['means']):,} peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB "
                  f"{time.time() - t0:.0f}s", flush=True)

    after = evaluate("after")
    snapshot("after")
    save_splat(params, os.path.join(out, "splat.ply"))
    np.save(os.path.join(out, "labels.npy"), state["label"].cpu().numpy().astype(np.int32))
    link = os.path.join(out, "export")
    if not os.path.exists(link):
        os.symlink(os.path.abspath(os.path.join(src, "export")), link)
    json.dump({"before": before, "after": after, "iters": a.iters, "width": W,
               "object_splats": [n_obj0, int((state["label"] > 0).sum())],
               "seconds": time.time() - t0, "peak_gb": torch.cuda.max_memory_allocated() / 1e9},
              open(os.path.join(out, "refine.json"), "w"), indent=1)
    print(f"done: {before['mean']:.2f} -> {after['mean']:.2f} dB held-out, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
