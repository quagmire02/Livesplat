# STATUS

Updated: Oct 30.

## Robotics lab — Oct 30/31
- Plan (user agreed: not only arms). A Lab bench with 4 tests, each over property ranges with a results card, a suggestion and an exportable report:
  - transport (mobile robots/trays)
  - gripper fit (jaws, suction)
  - grasp (arm + gripper)
  - camera (what a sensor sees)
- Scale (`measure/scale.py`): currently Muse typical sizes, ±11% spread. Calibrate from one known size ("the can is 15 cm tall"); A4/ArUco and LiDAR later. Uncalibrated tests also run at ±spread.
- Done:
  - `lab/transport.py` + `lab/objects.py` (3D-model hulls with a flat base, checked to stand). 3 known-answer tests.
  - table1 on wood: safe braking 1.75 m/s². The can tips first (0.22 g sideways); a rubber mat does not help, because tipping is the limit.
- Done: `lab/gripper.py` (jaws + suction) with 4 gripper files. table1:
  - Panda and Robotiq hold all three objects.
  - SO-100 jaw: the bottle holds in 22% of cases (grip force 12 N vs 16 N needed); the roll in 33% (58 mm vs a 60 mm opening: depends on scale).
  - Suction 30 mm: bottle front side OK; the can is too curved; paper is porous.
- Done: `lab/grasp.py` (arm grasp test over mass, friction and scale).
  - Quick run: Panda takes the can by the side and the roll from the top; the bottle slips. SO-100 is blocked or out of reach on this layout.
- Done: 3 more arms (ViperX 300 S, WidowX 250 S, xArm7) via `sim/robot_probe.py` (auto-drafted yaml).
- Done: `lab/camera.py` (D435/D455/ZED 2i/Femto Bolt/C920; visibility/occlusion, pixels, range, depth noise, auto placement, synthetic depth + masks).
  - D435 best at 0.45 m, 45° up (σz 0.5 mm). The D455's 0.52 m minimum range forces it to 0.8 m (about 50 px per object).
- Done: Lab tab (`viewer/lab_panel.py`): tests, cards, report download, camera frustum + render, 12-view dataset export. Typing "the can is 15 cm tall" in the main box calibrates.
- Full grasp sweep (27 cases per grasp: mass × friction × scale):
  - Panda: can side 85%, roll top 67% (scale-dependent), bottle 0% (slips).
  - xArm7: bottle top 100%, can 100%, roll top 33%.
  - SO-100: blocked or out of reach on this layout.
  - ViperX/WidowX: ~0% on everything. The Menagerie gripper kp is 200 N/m (~1 N grip), so `gripper.kp` 1500 was added to the robot yaml, but they still fail. The auto-drafted spec (TCP/open-close) is suspect, so they are marked "(unverified)" in the Lab tab.
- Fixed (Oct 31):
  - Trossen arms: robot joints are clamped into their ranges on reset. The WidowX fingers started at q=0, outside the 15-37 mm range, and jammed. All arms now pass the synthetic cube.
  - Side pre-grasp back-off now scales with object size (it was colliding with big objects).
  - Grip-force calibration (`sim/grip_calibrate.py`): stock squeeze was 1.3 N/finger for the Panda (real 70 N) and 402 N for the xArm7. All arms are now calibrated to force_n; the Panda value is from the data sheet, the others are estimates.
- Added: `lab/advisor.py` (what decides it, Muse grip vs planner, Muse-proposed changes simulated); Lab card shows these lines; "watch the arm" replays a success or failure in the live scan.
- First advisor run (Panda, uncalibrated grip): the uninformed Muse grip was worse than the planner (can 22% vs 100%). Given the planner's facts it matched (89%, 100%). Pads on the bottle: 0 -> 11%.
- Next: the calibrated 5-arm sweep with the advisor (running); click through the Lab tab.

## CONSUMER SIDE LOCKED — Oct 30 (tag `consumer-lock`)
- Frozen: the one-box Ask/Tell (explain / similar / guide / 3D model / 3D look / commands / Director), scene clean-up, fix the scan, Muse names. Only bug fixes from here.
- 3D look ("3D look" button, or "use the 3d model look" / "back to the scanned look"): the Hunyuan model replaces the scan splats. A separate `/look<k>` node follows the physics pose every frame (`qmul(pose, Q_YUP_TO_ZUP)`, tested in `tests/test_look_quat.py`), so drag, push, drop and scale keep working. Recolouring only affects the scanned look.
- Next: robotics lab (real-to-sim for university labs: grasp/part tests across property ranges, results + suggestions; LiDAR importer when the user has iPhone footage).

## 3D models + Muse names — Oct 30 (late night)
- Names: `agent.properties` now sends 3 views and asks for brand, specific type and visible text. `--names-only` refreshes names.json without touching mass/friction/scale.
  - table1 is now "cetaphil moisturizing lotion pump bottle", "gillette aerosol deodorant spray" (alias "shield"), "toilet paper roll". The cut-outs show "Gillette" on the can.
- Similar search uses the name plus the visible text. The panel says "Possible matches" and sits beside an explanation.
- `viewer/route.py`: one message can trigger several modes (explain + similar). The object can be named in the text (`find_object`). 25 routing tests.
- 3D models (`viewer/model3d.py` prep/paint + `segment/hy3d_shape.py`):
  - Muse picks the front photo; the real frames nearest -90/180/+90 deg are cut out with SAM 3.1 and fed to Hunyuan3D-2mv turbo (own venv `~/venvs/hy3d`).
  - The mesh is aligned to the scan (complete3d.align) and every vertex is coloured from the most head-on real photo, z-buffered against all objects.
  - About 13 s per object, peak 3.1 GB GPU. The Cetaphil label is readable. Much better than TripoSR.
  - Gotchas: the stock loader builds fp32 on CPU (about 10 GB), which OOM-killed WSL, so the runner loads fp16 per part. transformers 4.48.3 + diffusers 0.32.2 are pinned in the hy3d venv. CPU offload needs `components`, `_exclude_from_cpu_offload` and `device=cuda` set by hand. `enable_flashvdm(replace_vae=False)`.
  - Don't run the viewer and Hunyuan at the same time (8 GB WSL RAM).
  - Viewer: the "3D model" button (and phrases like "make a 3D model of this") shows `gen/model_<id>/model.glb` beside the object (viser add_glb) and sends it as a download. Pressing it again hides it; "put back" removes it. If the model is missing, it tells the user to run `python -m viewer.model3d build <run> <id>` with the viewer closed; build refuses while the viewer runs.
  - Comparison images are in `.preview3d/compare.jpg` (git-ignored).
- Dependencies: hy3d venv (torch 2.6 cu124, diffusers, transformers, accelerate, pymeshlab), separate from LiveSplat. Reason: user-facing 3D models from scans.

## Consumer refinement — Oct 30 (night)
- User plan: refine and lock the consumer side (Ask/Tell, explain, scene clean-up, fix the scan), then move to the robotics lab (the main dark horse).
- Ask/Tell:
  - The "price" button is replaced by "similar": the same product's links, 8 pictures and similar products (DuckDuckGo text + images, Muse minimal), about 9 s.
  - Routing moved to `viewer/route.py` with 21 test phrasings.
  - Muse tool `find_similar` added.
- 3D model check: the TripoSR single-image meshes (`gen/obj_k.obj`) were viewed. The bottle is lumpy, the can bent, the roll failed. Not good enough for a user feature.
  - Proposal: Hunyuan3D-2mv (multi-view shape from real scan photos) plus texture projected from the real photos. Waiting on the user's OK to download the weights.
- iPhone LiDAR: MapAnything accepts depth and poses, so a Stray Scanner / Record3D importer would give metric scale. Needs a LiDAR iPhone/iPad to test.
- `muse-wonder-1.0` is on our API: Meta's frontier text/reasoning model (video input, $4/$20 per 1M tokens). It is not 3D.

## Explain fixes + Guide mode — Oct 30 (evening)
- Muse `reasoning_effort` works on the Meta API: default 6.8 s, low 4.0 s, minimal 1.4 s (`client.spark_chat(extra=...)`). Planner and Explain/Guide use low.
- Explain:
  - Muse picks the front photo (`front_photo`), which now drives the diagram, the turn-to-face and SAM.
  - SAM runs in parallel with a second phrasing per part (`sam_alt`). The most confident mask on the object wins; a "part" mask covering the whole object is rejected.
  - Measured 5–13 s, down from about 30.
  - The object floats in view, never inside the table. The voice is at 0.9 volume and prefers natural/neural/Google English voices. The price panel moves beside the Explain panel.
- Typed questions skip routing:
  - Selected object + "what is this / how do I use / explain" goes to Explain.
  - "How do I… / how to… / step by step" goes to Guide.
  - The `guide` tool also exists for Muse.
- Guide (`viewer/explain.py: guide`): one photo per object goes to Muse, which returns steps that point at objects plus "not on the table" items. The viewer highlights and labels each object while its step is spoken. Headless test: 5 s, sensible steps.
- Table view detail: the same SH view colour as the objects, for background splats within 1.3× the table radius, refreshed when the camera settles. Startup checked; the visual result has not been checked in a browser.

## Simple UI + Explain mode — Oct 30 (later)
- User feedback: too many sliders/options; the Director button felt odd and slow. Now:
  - The main panel is one "ask or tell" box, a send button, and buttons for the selected object (explain, price, put back). Everything else is in a collapsed "More tools".
  - The box routes everything through Muse: plain commands, `plan_goal` (Director), `explain`. There is no Director button.
  - Director runs 2 rounds max and stops at score ≥ 8; progress is one plain-language notification.
- Explain mode (`viewer/explain.py`): Muse vision chooses parts and narration, SAM 3.1 finds each part in close-up crops, and the masks become 3D labels on the object's splats. The object floats in front of the camera, turned to the photographed side. The browser reads the narration aloud (Web Speech, via an img onerror in add_html) while each part is highlighted. A photo diagram with outlines sits in the left panel.
  - CLI-verified on the spray can: cap, body and label, about $0.05.
  - Not yet clicked through in the browser; speech depends on the browser allowing it.
- The budget ledger had a NUL tail from an unclean WSL shutdown. It was cleaned (backup `ledger.jsonl.bak-*`), and `spent()` now skips unreadable lines with a warning.

## Director mode + market search — Oct 30
- Verified on table1 via Baseten (each goal about $0.05–0.10):
  - "knock the roll off the table": met in round 1.
  - "bottle next to the can, not touching, standing": met (1 cm gap) after `slide` was added.
  - "line up all three in a row": 8/10 (gaps fine, but a triangle, not a line).
- Physics bug fixed: the table/floor friction (0.6/0.8) overrode every object's friction, because MuJoCo takes the max of the two. Both are now 0.02, so `set_friction` works (bottle slides 1.22 at mu 0.1 vs 0.08 at mu 0.8).
- New tool `slide(obj, direction, distance | target, gap)`: hand-guided steady move that stays upright; with a target it stops just short of it. `push` remains a flick. Tests in `tests/test_slide.py`.
- Director: retries once with double tokens if a reply spent all its tokens on reasoning.
- The viewer starts cleanly. The Director button and the market search are not yet clicked through in the browser.
- `agent/director.py`: goal → Muse proposes 6 plans → each runs headless in MuJoCo → Muse scores the measured outcomes and revises (3 rounds) → best plan plays live. Viewer: Play > "plan it · Director" (planner must be meta/baseten). Logs go to `runs/<scene>/director/*.json`. CLI: `python -m agent.director <run> "<goal>" --provider baseten`.
- `backends/search.py`: free product search. SearXNG (`LIVESPLAT_SEARXNG_URL`) or DuckDuckGo (`ddgs`, no key) → Muse picks listings; prices and urls are checked against the result text. Viewer: Object > "Search the market" (needs a loaded analysis card).
- Budget: total 140 USD, per call 4 USD (user decision). Muse is the default planner when allowed.
- Works: Baseten connection (confirmed by the user).
- `ddgs` is installed (`uv pip`; the venv has no pip) and returns Amazon/Target results with no key.
- Next step: click through Director and market search in the viewer. Absolute placement ("row", "centre") needs a slide to an xy point.

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
- ddgs (to install): free keyless web search for market lookup; Google's search API no longer takes new users.

## SAM-grounded reference panel (2026-10-29)
- Select -> Isolate with SAM 3.1 (cached, $0) -> Muse analyses masked crops + diagnosis overlays -> card with scan issues, capture guidance, suggested fix (preview/apply via Repair). Live Muse round for PROMPT_SAM not yet approved.

- Muse PROMPT_SAM validated live on table1 (approved round, \.035 for 3 objects): names, visible text (Cetaphil, SHIELD), per-view scan issues, capture guidance, repair advice remove_attachments for all three. Total spend \.328.

- Baseten (Muse Spark Open) validated: tool calls and vision work; planner option baseten (multi-step stack command works). Spark Open sends ids as numbers, coerced to text.

- Fix semantics corrected: SAM spill splats in the air are ERASED (fuzz), only table-band splats stay on the table (pinning fuzz left a ghost at the original spot). UI regrouped into 4 tabs; bottom toast for command feedback.
