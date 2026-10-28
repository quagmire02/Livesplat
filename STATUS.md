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

## Day 2 chunks (done)
- A: base band joins object; glossy mirror image = label -k, shown only at rest.
- B: physics/world.py (MuJoCo) + viewer GUI; settle test passes; push tips the can.
- C: recon/build.py presets with VRAM/RAM fallback (dry-run verified).
- D: recon/refine_objects.py works (lazy native crops, frozen bg). No gain on a 1568 px video (expected); auto-skipped there.

## Day 2 later chunks (done)
- Agent: 16 tools, local rule parser, Spark planner (not called yet), controller; 15 tests pass.
- Hole fill under moved objects; reflections follow upright objects (shiny can went pale when hidden).
- Hull: outlier filter + flat footprint from all low Gaussians (thin can tipped at rest before).
- Scale is ~1.5-2x too large (table 97 cm across); needs A4 capture.

## Day 2 evening (done)
- Editor selection (click + gizmo, throw on release), full reset, compact panel, tolerant parser (28 tests).
- Spark round 1 (approved): names + props for 3 objects ($0.011) - names right; toilet roll mass 0.02 kg is wrong (empty-tube guess). Planner 4/5 on free wording ($0.022); miss = looked up objects then stopped -> multi-round tool loop written, NOT yet tested (no API until base features done).
- Total API spend $0.0511.

## Product direction
- Chosen: real-to-sim for robotics. Branch real-to-sim. Plan: docs/REAL_TO_SIM.md.

## Real-to-sim (branch real-to-sim)
- T1-T9 done; T10 (convex decomposition) not done. Results: sim/README.md.
- Fixes: full-room background (14 -> 25 dB); primitive collision stacks (convex-mesh grasps slipped); aim at slice centre; 2.5 cm grasp depth (palm hit the top at 3.5-5 cm).
- Next: side grasp (bottle); calibrated capture (A4) for units_per_metre; demo video.

## BYO robots (done: B1, B2)
- robot.yaml spec; Panda and SO-100 (5-DoF, jaw, table-clamped). Metric export (--scale). Synthetic known-answer scene. 32 tests pass.
- Product focus recorded: core = scan, tag, move; Simulation mode is optional.

## Blocked / needs user
- Better captures with an A4 sheet (for metric accuracy).

## Dependencies added
- viser, mujoco, trimesh, scikit-learn, sam2, meta-sam-parser (SAM 3.1 decoder), pycolmap (global SfM, CPU).
