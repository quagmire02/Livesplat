# recon/ — stage 0 (reconstruction) + quality presets
Files:
- build.py — ONE COMMAND: `python -m recon.build <video> <name> --quality low|medium|high [--dry-run]`. Probes video, free VRAM/RAM; lowers preset until estimates fit; retries lower on OOM. Writes runs/<name>/{build.json, build.log}.
- run_thrift.sh — stage 1 (cloud) via ThriftSplat venv.
- sfm_and_train.sh — global SfM (pycolmap, LiveSplat venv) -> train_gsplat (iters, width, cap) -> export. Skips SfM if done.
- export_thrift.py — <run>/export/cameras.json + keyframes. c2w includes pose_delta.
- refine.sh / refine_objects.py — object-focused refinement ("zoom"): only object Gaussians train + densify on native-res crops; background frozen, no pruning; labels carried through densify; held-out crop PSNR before/after. Frames load lazily (crops only in RAM).
Presets: low fps2/15k/1554/1.8M; medium fps3/20k/1554/2.0M/refine 3k; high fps4/25k/<=2072/2.2M/refine 6k.
Refinement only runs when the video is >= 1.25x wider than the training width (else no new pixels).
Status: table1 low done. Refinement on table1 (1568 px video): no gain (-0.03 dB no-densify, -0.9 dB with densify, 1k iters) -> correctly auto-skipped. Needs a 4K capture to show gain.
