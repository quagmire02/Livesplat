# recon/ — stage 0 (reconstruction)
Purpose: call ThriftSplat (external, own venv) and export its run to LiveSplat format.
Files:
- run_thrift.sh — stage 1 (cloud) via ThriftSplat venv.
- sfm_and_train.sh — global SfM (pycolmap, LiveSplat venv) -> train_gsplat -> export. Skips SfM if done.
- export_thrift.py — run with $THRIFT_PY. Writes <run>/export/cameras.json + keyframes/*.jpg. c2w includes pose_delta.
Inputs: ThriftSplat run dir (scene/, splat/, splat.ply). Outputs: <run>/export/.
Status: export works (orbit_sfm, 113 frames, 13 s).
