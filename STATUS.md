# STATUS

Updated: Oct 29.

## Documentation handoff — Oct 29
- Added root CHANGE_NOTES.md and short folder notes in viewer, agent, backends, backends/meta, recon, scene_io, segment, tests and docs. Each distinguishes implemented work from pending features; existing folder indexes link to their notes.
- Outstanding work includes visual fidelity comparison, painting and completion, approved SAM/Muse validation, online reference search, export integration and medium-priority metric calibration. General-scene work remains deferred.
- Documentation-only pass; no model calls, commits or pushes.

## Object references / Muse analysis — Oct 29 (uncommitted)
- Selected objects show a responsive left sidebar with up to three source-video crops chosen for projected size and viewpoint diversity. Original capture coordinates are used even after moving/scaling the object. References tab offers explicit cache-only analysis and approved live Muse analysis; results include suggested identity, visible text, evidence, uncertainty, capture guidance and search terms. Name confirmation is explicit and undoable; physics properties stay unchanged.
- Reuses existing object segmentation; no new SAM calls. Crops may contain neighbouring objects and are not pixel-perfect masks. Model prompt explicitly forbids claiming online matches or diagnosing splat holes from source photos. Marketplace/image search, fresh SAM refinement and rendered-view diagnosis remain future work.
- Browser verified three thumbnails at left x=12, live button disabled, and cache miss returns without network. Model output validation and distinct-view selection tests pass. No external API calls or added dependencies; no commits/pushes. Review host localhost:8001 runs with API access disabled.

## Repair workspace first slice — Oct 29 (uncommitted)
- New Repair tab: choose object and object/contact layer, isolate, enable click brush, preview front-depth selection, erase/restore or assign to object/table. Reassignment requires original pose and scale. 30-step repair undo/redo; explicit atomic visual_repairs.npz save/load, checked against geometry hashes. Original scan/collision geometry untouched; visual repair overlays follow the object or remain at capture position.
- Validation: 50 focused offline tests passed, 1 skipped. Real-scene browser opened Repair and isolated the lotion bottle; software WebGL only. API access disabled. No dependencies, commits or pushes.
- Browser brush check selected and erased 246 lotion-bottle splats, then invoked repair undo. Latest review host is localhost:8001; port 8000 is preserved.
- Scope remaining: source-frame rendering comparison, colour painting, continuous brush strokes, background-to-object recovery beyond the contact layer, small-hole completion and guided recapture. No geometry completion or improved reconstruction fidelity is claimed by this editing slice. Repair overlays are viewer-only and do not export to simulation; saves load on explicit request.

## Splat contact / appearance review — Oct 29 (uncommitted)
- Viewer now separates low table-like contact Gaussians from movable object bodies. Captured contact and reflection layers remain anchored, and hide when the object moves/rotates/scales/is removed. Table fill uses the removed scan footprint rather than a potentially unrelated generated collision mesh. Original PLY, labels and collision shapes are unchanged.
- Restores degree-3 learned object colour terms previously discarded by the RGB-only viewer. Explore has a view-dependent object detail toggle for comparison. Refresh is debounced after camera motion; clients share the latest camera shading. This restores existing appearance, not unseen geometry or super-resolution.
- Real table1 partition: 16,521 contact Gaussians anchored across three objects; all three have fill patches. Saved-layout geometry fingerprints retain their original meaning. Existing generated fill-colour caches are bypassed for changed patch geometry.
- Validation: 48 focused offline tests passed, 1 skipped, including numerical comparison against gsplat SH evaluation. Review host restarted at http://localhost:8001 with model API access disabled; port 8000 preserved.
- Limits: contact separation is a geometric heuristic; occluded undersides remain missing and exposed table is interpolated. Existing crop-refinement record showed no gain (27.3985 to 27.3710 dB); digital zoom cannot recover uncaptured pixels. Simulation-export splats still use their original labels. No new dependencies, model/API calls, commits or pushes.
- Next: user visual review of moving each object and colour-toggle comparison before further work.
- Browser validation: selected lotion bottle, placed it at X=0.25, inspected rendered screenshot, restored original position. No browser/server errors. Headless rendering used software WebGL; remaining shell-like geometry is visible and not claimed fixed.

## Editor baseline pass — Oct 28, codex/new-features (uncommitted)
- User direction: ordinary users explore/rearrange; labs run physics/robotics; measurement benefits both. Concrete scope and next steps: docs/EDITOR_PLAN.md. Stop after this pass for review.
- Explore / Inspect / Clean up / Robotics tabs. Arrange starts paused; Experiment runs physics; optional throw-on-release. Object list, names/properties, precise upright placement, absolute resize, restore/remove, comparison, camera presets.
- Compact 30-step undo/redo; atomic editor_state.json save/load with geometry validation; CSV inventory download through the web host. --load-layout restores a saved layout on startup, paused. Background cleanup has separate history. Object additions clear layout history.
- Fixed numeric object IDs being used as drop heights, zero-size parser crash, malformed tool validation, yaw counted as tipping, and prediction state leakage. Predictions use a separate MuJoCo data instance.
- API requests now require LIVESPLAT_ALLOW_META=1 after cache lookup; explicit cache-only mode never falls through to network. Build tagging is opt-in (--tag), completion masks are cache-only, nullable Muse tool messages work, retries recheck budget. Live access stays disabled in this session. The $150 hackathon credit is not spend authorization.
- Validation: 46 focused offline tests passed, 1 real-scene test skipped. Mocked model responses only; sockets blocked. Browser checks on isolated local host passed: selection, properties, undo/redo, save/load, CSV download, precise placement, resize/undo and separation comparison. Viewed rendered scene and inspector screenshots. Headless browser used software WebGL; no performance benchmark claimed.
- No external API calls, new dependencies, commits or pushes. Existing robotics work was not expanded in this pass.
- Review host: http://localhost:8001 on the real table1 capture, latest code, API access disabled. The existing http://localhost:8000 session was preserved because automated save/restart could not be completed reliably in that active browser session. Temporary isolated browser-test data was removed; scan assets were preserved.
- Medium priority next: user-entered measured dimensions / global calibration, propagated into physics and robot exports. Current inspector/CSV dimensions remain scene units; saved editor layouts are not yet used by sim.export.

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

## SAM-grounded reference panel (2026-10-29)
- Select -> Isolate with SAM 3.1 (cached, $0) -> Muse analyses masked crops + diagnosis overlays -> card with scan issues, capture guidance, suggested fix (preview/apply via Repair). Live Muse round for PROMPT_SAM not yet approved.

- Muse PROMPT_SAM validated live on table1 (approved round, \.035 for 3 objects): names, visible text (Cetaphil, SHIELD), per-view scan issues, capture guidance, repair advice remove_attachments for all three. Total spend \.328.

- Baseten (Muse Spark Open) validated: tool calls and vision work; planner option baseten (multi-step stack command works). Spark Open sends ids as numbers, coerced to text.
