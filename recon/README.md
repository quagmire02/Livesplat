# recon/ — video to splat (stage 0)
Purpose: Make a trained splat from a video. Use ThriftSplat in its own venv.
Files:
- build.py — One command. `python -m recon.build <video> <name> --quality low|medium|high [--dry-run]`. It checks VRAM and RAM. It lowers the preset if the estimate does not fit. It tries again lower after out-of-memory.
- run_thrift.sh — Stage 1: video to point cloud and poses.
- sfm_and_train.sh — Global SfM (pycolmap), then splat training, then export. It skips SfM if done.
- export_thrift.py — Writes export/cameras.json and keyframes. Poses include pose_delta.
- refine.sh, refine_objects.py — Object refinement on native-resolution crops. Background stays frozen. Use only if the video is wider than the training width.
Inputs: video. Outputs: runs/<name>/{thrift, thrift_sfm, build.json, build.log}.
Status: Works on table1 (low). Refinement gives no gain on a 1568 px video. A 4K video is necessary to show a gain.
Build is local by default. Use --tag to request Spark/SAM tagging; cached results are free to replay, while cache misses require an explicitly approved API round (LIVESPLAT_ALLOW_META=1). --no-tag remains supported.

- [CHANGE_NOTES.md](CHANGE_NOTES.md) — Short implementation summary and remaining gaps (Oct 29).
