# scene_io/ — load, save, conventions
Files:
- splat.py — PLY load/save (raw values on disk), subset, activate, covariances, transform (wxyz).
- thrift.py — load_cameras (export/cameras.json), load_splat, project (OpenCV).
Rules: See CLAUDE.md section 3. ThriftSplat scales K linearly. The offset is 0.71 px at 1036 px. Training used the same K, so the offset is safe.
Status: Works. Camera test passes.
