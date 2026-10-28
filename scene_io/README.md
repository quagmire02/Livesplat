# scene_io/ — load/save + conventions
Files:
- splat.py — PLY raw load/save, subset, activate, covariances, transform (wxyz).
- thrift.py — load_cameras (export/cameras.json), load_splat, project (OpenCV).
Conventions: see CLAUDE.md §3. K from ThriftSplat is linear-scaled (no half-pixel shift); 0.71 px offset at 1036 px, consistent with training.
Status: written; camera test passes.
