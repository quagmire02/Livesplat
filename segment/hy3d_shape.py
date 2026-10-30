"""Hunyuan3D-2mv shape from real multi-view cut-outs (front/left/back/right.png with alpha) -> shape.glb.

Runs in its OWN venv (~/venvs/hy3d; Hunyuan needs newer transformers than TripoSR in the LiveSplat venv):
  ~/venvs/hy3d/bin/python segment/hy3d_shape.py <run>/gen/model_<id> [--low-vram]
Code: ~/livesplat_data/models/Hunyuan3D-2 (Tencent, external). Weights: tencent/Hunyuan3D-2mv mv-turbo (HF cache).
Inputs/outputs are written by viewer.model3d (prep before, paint after).
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image

import os  # noqa: E402

MODELS = Path.home() / "livesplat_data" / "models"
SUB = "hunyuan3d-dit-v2-mv-turbo"


def _local_weights():
    """hy3dgen loads <HY3DGEN_MODELS>/<repo>/<subfolder> if it exists; otherwise it downloads the whole subfolder
    (incl. a second 4.9 GB .ckpt copy). Link the already-downloaded safetensors snapshot there instead."""
    from huggingface_hub import snapshot_download
    snap = Path(snapshot_download("tencent/Hunyuan3D-2mv", allow_patterns=[f"{SUB}/config.yaml",
                                                                          f"{SUB}/model.fp16.safetensors"]))
    link = MODELS / "hy3dgen" / "tencent" / "Hunyuan3D-2mv" / SUB
    if not link.exists():
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(snap / SUB)
    os.environ["HY3DGEN_MODELS"] = str(MODELS / "hy3dgen")


_local_weights()
sys.path.insert(0, str(MODELS / "Hunyuan3D-2"))
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline  # noqa: E402


def _low_memory_from_single_file(cls, ckpt_path, config_path, device="cuda", dtype=torch.float16,
                                 use_safetensors=None, **kwargs):
    """Replacement loader for 8 GB RAM: the stock one reads all 4.9 GB, then builds every part in fp32 on the CPU
    (~10 GB) and gets OOM-killed. Here each part is built directly in fp16 and filled from the safetensors file
    one part at a time (peak ~ one part twice)."""
    import safetensors
    import yaml
    from hy3dgen.shapegen.pipelines import instantiate_from_config
    cfg = yaml.safe_load(open(config_path))
    path = ckpt_path.replace(".ckpt", ".safetensors")
    parts = {}
    with safetensors.safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        torch.set_default_dtype(torch.float16)
        try:
            for name, strict in (("model", True), ("vae", False), ("conditioner", True)):
                m = instantiate_from_config(cfg[name])
                sd = {k[len(name) + 1:]: f.get_tensor(k) for k in keys if k.split(".")[0] == name}
                if sd:
                    m.load_state_dict(sd, strict=strict)
                del sd
                parts[name] = m
        finally:
            torch.set_default_dtype(torch.float32)
    return cls(vae=parts["vae"], model=parts["model"], conditioner=parts["conditioner"],
               scheduler=instantiate_from_config(cfg["scheduler"]),
               image_processor=instantiate_from_config(cfg["image_processor"]), device=device, dtype=dtype, **kwargs)


Hunyuan3DDiTFlowMatchingPipeline.from_single_file = classmethod(_low_memory_from_single_file)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="one or more gen/model_<id> folders")
    ap.add_argument("--all-on-gpu", action="store_true", help="no CPU offload (needs > 6 GB VRAM)")
    ap.add_argument("--octree", type=int, default=256)
    a = ap.parse_args()
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2mv", subfolder=SUB, variant="fp16", use_safetensors=True,
        device="cuda" if a.all_on_gpu else "cpu")
    if not a.all_on_gpu:                                  # parts take turns on the 6 GB GPU
        pipe.components = {"conditioner": pipe.conditioner, "model": pipe.model, "vae": pipe.vae}
        pipe._exclude_from_cpu_offload = []
        pipe.enable_model_cpu_offload()
        pipe.device = torch.device("cuda")               # latents/inputs on the GPU; hooks bring the weights
    pipe.enable_flashvdm(replace_vae=False)               # keep the mv-turbo VAE (no extra download)
    for d in map(Path, a.dirs):
        views = {v: Image.open(d / f"{v}.png").convert("RGBA")
                 for v in ("front", "left", "back", "right") if (d / f"{v}.png").exists()}
        t = time.time()
        mesh = pipe(image=views, num_inference_steps=5, octree_resolution=a.octree, num_chunks=20000,
                    generator=torch.manual_seed(12345), output_type="trimesh")[0]
        mesh.export(d / "shape.glb")
        print(f"{d.name}: {sorted(views)} -> {len(mesh.vertices):,} verts in {time.time() - t:.1f}s "
              f"(peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB)", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
