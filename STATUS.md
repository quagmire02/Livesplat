# STATUS

Updated: Oct 28 (day 1, evening).

## Works
- venv ~/venvs/livesplat (torch 2.6 cu124, mujoco 3.17, viser 1.1.1, sam2, meta-sam-parser, pycolmap 4.2.1).
- recon/run_thrift.sh (stage 1 cloud), recon/sfm_and_train.sh (global SfM -> train -> export), recon/export_thrift.py.
- Meta API smoke PASS, total $0.0176. SAM 3.1 masks decode correctly (backends/meta/sam_parse.py). Spark vision JSON + tool calls OK.
- segment/plane.py finds the table on table1 stage-1 cloud: radius 0.40 (scene units).

## Day-1 spikes
| # | Spike | Result |
|---|---|---|
| 1 | Meta API smoke | PASS ($0.0176) |
| 2 | Viser nodes | PASS |
| 3 | Fuzz after move | PASS after region growing (small grey footprint remains -> hole fill) |
| 4 | SAM 2.1 small | PASS (0.41 s/frame, 1.6 GB) |
| 5 | Install | PASS |
| 6 | Camera convention | PASS (0.71 px) |

## table1 (first real capture)
- Video 1568x736, 36.7 s, round wooden table, 3 objects (pump lotion bottle, blue spray can, toilet roll). No A4 reference. Glossy table top.
- Stage 1: 74 frames, 4.0 min, parallax 0.040 (good).
- SfM: 74/74 registered, 0.97 px.
- Training: running (15000 iters, width 1554). First try failed: ninja not on PATH (fixed in sfm_and_train.sh).

## Lessons
- Spark 1.3 is a reasoning model: max_tokens >= 1500 or content is null.
- SAM 3.1 prompts: simple nouns ("bottle"); "spray can" gave 0 matches. Empty output = 0 matches, not an error.
- SAM 3.1 video = $0.20/1000 frames: the whole clip costs ~$0.22. May replace the SAM 2 hybrid.
- Table detection: nearest-plane-to-look-at picked the FLOOR. Fixed: up = camera-orbit normal, table = densest height in a cylinder around the look-at axis, extent = radial density edge (DBSCAN components chained into walls).
- WSL quoting: put multi-line commands in a script file; run via PowerShell `wsl.exe -- bash -l <file>`.

## table1 segmentation (done)
- Splat: 1.47M Gaussians, 54 min train, peak 2.74 GB. Table radius 0.487 (scene units).
- 3 objects found = 3 real objects: lotion bottle (110k G), spray can (69k), toilet roll (68k).
- Fixes: occupancy flood fill for table extent (glossy top is sparse; objects hide the centre); fragment filter; grow_labels (low-opacity Gaussians were left behind as a ghost column).
- viewer/app.py: bg + per-object nodes in table frame, manual sliders; bg cropped to 1.6x table radius.

## Next (Oct 29)
- physics/: MuJoCo world from object hulls, step loop driving Viser nodes, push/drop/reset.
- measure/: OBB + volume (needs A4 capture for metric check).
- Viewer start camera looks from a low angle; fix.

## Blocked / needs user
- Better captures with an A4 sheet (for metric accuracy).

## Dependencies added
- viser, mujoco, trimesh, scikit-learn, sam2, meta-sam-parser (SAM 3.1 decoder), pycolmap (global SfM, CPU).
