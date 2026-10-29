# scene_io/ — load, save, conventions
Files:
- appearance.py — CPU evaluation of degree 0–3 learned spherical harmonics, with camera directions in the original reconstruction frame. The viewer refreshes object RGB after camera motion settles; no generated detail or API calls.
- splat.py — PLY load/save (raw values on disk), subset, activate, covariances, transform (wxyz).
- thrift.py — load_cameras (export/cameras.json), load_splat, project (OpenCV).
Rules: See CLAUDE.md section 3. ThriftSplat scales K linearly. The offset is 0.71 px at 1036 px. Training used the same K, so the offset is safe.
Status: Works. Camera test passes.

- [CHANGE_NOTES.md](CHANGE_NOTES.md) — Short implementation summary and remaining gaps (Oct 29).
