# STATUS

Updated: Oct 28 (day 1).

## Works
- Repo skeleton (local git, not pushed), CLAUDE.md, folder READMEs.
- venv ~/venvs/livesplat: torch 2.6.0+cu124 (CUDA OK), mujoco 3.17.0, viser 1.1.1, sam2 (no CUDA ext; harmless warning), SAM 2.1 small ckpt in ~/livesplat_data/models.
- recon/export_thrift.py on orbit_sfm (113 frames, 1036x588, 13 s).

## Day-1 spikes
| # | Spike | Result |
|---|---|---|
| 1 | Meta API smoke (cap $1) | NOT RUN. Needs MODEL_API_KEY in WSL + user go |
| 2 | Viser nodes | PASS. Node pose moves Gaussians rigidly (shapes rotate too). 600k Gaussians, smooth |
| 3 | Fuzz after object moves | PENDING. Needs a tabletop scene |
| 4 | SAM 2.1 small | PASS. 40 frames 1036x588: 0.41 s/frame, peak 1.60 GB |
| 5 | Install | PASS (torch download needed UV_HTTP_TIMEOUT=900) |
| 6 | Camera convention | PASS. 0.71 px constant half-pixel offset (ThriftSplat linear K scaling; consistent with training) |

## Notes
- Viser start camera must be set (client.camera.look_at), else view is empty/far.
- Splats look white from far away (floaters); fine up close.
- WSL: background servers must run via the tool's background mode; `nohup ... &` dies with the wsl.exe session. Use pkill -f "[s]pikes..." (bracket stops self-match).

## Next (Oct 29)
- segment/plane.py + cluster.py on the first tabletop scene; objects as Viser nodes.

## Blocked / needs user
- Tabletop capture (3 scenes, CLAUDE.md §2), then ThriftSplat run + export.
- MODEL_API_KEY for spike 1.

## Dependencies added
- viser, mujoco, trimesh, scikit-learn, sam2 (CLAUDE.md §5–8).
