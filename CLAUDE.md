# LiveSplat

Film a tabletop with a phone. Get a photoreal 3D scene. Measure the objects. Push, drop and stack them with physics. Control the scene with text commands (voice is optional).

Meta Global AI Developer Hackathon 2026, Indie track, theme "Push the Frontier".
Submission: demo video + public repo. Deadline: **Nov 4, 9:00 PM ET** (Nov 5, 6:30 AM IST).

---

## 1. Hard rules (read first)

1. **Muse models are the default (user decision, Oct 30).** About $145 of credit, 4 days left. Use the meta/baseten backends rather than the local ones. Viewer: run with `LIVESPLAT_ALLOW_META=1` (and `LIVESPLAT_ALLOW_BASETEN=1`), and the planner then defaults to `meta`.
2. **Budget guard.** `backends/meta/budget.py` writes every paid call to `.cache/api/ledger.jsonl` with its estimated cost. It refuses a call if the total would go above `LIVESPLAT_BUDGET_USD` (default **140**), or if a single call's estimate is above `LIVESPLAT_MAX_CALL_USD` (default **4**). Do not bypass it.
3. **Every API response goes to the disk cache** (`.cache/api/`, keyed by a hash of the request). Read the cache before any call.
4. **Do not copy ThriftSplat code into this repo.** Call ThriftSplat as a separate program (stage 0) in its own venv. It is the reconstruction front end.
5. **6 GB GPU.** Do not run SAM 2 and splat training at the same time. Release GPU memory between stages.
6. **Each folder has a `README.md` index.** Read the index before you open files in that folder. Update the index when you add, remove or rename a file, or change inputs/outputs.
7. **Update `STATUS.md`** at the end of each work block: what works, what is broken, and the next step.
8. Keep diffs small. Do not reformat files you do not change. Do not add a dependency without a one-line reason in `STATUS.md`.
9. Do not write new code on Nov 4. Nov 4 is for the demo video and README only.
10. **No Claude attribution.** Do not add Co-Authored-By or "Generated with" lines to commits or PRs. The user is the only contributor.
11. Remote: github.com/quagmire02/Livesplat (branches main, real-to-sim).

---

## 2. Scope

**In scope:** one table, 4–6 rigid matte objects placed **at least 3 cm apart**, and one reference object of known size (A4 sheet or ArUco marker). Capture: an orbit or arc of 20–40 s.

**Out of scope:** full rooms, drone footage, glass/mirror objects, cloth/fluids, live reconstruction, Muse Image inpainting.

**Product focus (user decision):** The core product is for normal users: scan any tabletop, identify and tag the objects, move them freely (viewer + agent). Robot arms, sim export and grasp checks are an optional **Simulation mode** inside the product, not the product itself. Every new feature must keep the core flow simple.

**Consumer side LOCKED (user decision, Oct 30):** Ask/Tell (explain, similar/possible matches, guide, 3D model, 3D look), scene clean-up and fix-the-scan are frozen at git tag `consumer-lock`. Only bug fixes from now on. New work goes to the **robotics lab** (the main focus): real-to-sim for university robot labs, grasp/part tests with results and suggestions.

**Novelty:** none claimed. LiveSplat integrates known methods. Do not write "first" or "novel" in the README or the demo.

---

## 3. Conventions (most bugs come from here)

| Item | Convention |
|---|---|
| Units | Metres, kilograms, seconds |
| World frame | Z up. Origin on the table plane. Set by `segment/plane.py` |
| Camera pose | camera-to-world 4x4, **OpenCV** axes (x right, y down, z forward) |
| Foreign formats | Convert at load time, in `scene_io/` only |
| Quaternion | **wxyz** everywhere (gsplat, MuJoCo and Viser use wxyz). scipy uses xyzw: convert at the call site |
| PLY on disk | **Raw** 3DGS parameters: log scales, logit opacity, SH coefficients. Activate in memory only. Never write activated values |
| Object IDs | int, 0 = background/table, 1..N = objects |
| Array dtypes | float32 for geometry, uint8 for images |

Any code that crosses a convention boundary must have a unit test.

---

## 4. Pipeline (stages and artifacts)

Each stage reads files and writes files to `$LIVESPLAT_DATA/runs/<scene>/`. Re-run only the stage that failed.

| # | Stage | Input | Output |
|---|---|---|---|
| 0 | Reconstruct (ThriftSplat, external) | `clip.mp4` | `thrift/` = ThriftSplat output (`splat.ply`, `scene/scene.json`, `scene/frame_*.npz`) |
| 1 | Plane + crop (`segment/plane.py`) | `thrift/` | `align.json` (world transform), cropped Gaussians |
| 2 | Cluster (`segment/cluster.py`) | cropped Gaussians | `labels_geo.npy` (per Gaussian) |
| 3 | SAM vote (`segment/sam_vote.py`) | keyframes + labels | `labels.npy`, `names.json` |
| 4 | Scale + measure (`measure/`) | `labels.npy`, reference object | `measure.json`, scale factor |
| 5 | Properties (`agent/properties.py`) | object crops | `props.json` (material, density, mass, friction) |
| 6 | Physics world (`physics/`) | labels, measure, props | `world.xml` (MuJoCo) |
| 7 | Viewer + agent (`viewer/`, `agent/`) | all above | interactive app |

- **Stage 0:** `recon/run_thrift.sh` calls ThriftSplat with its own venv (`THRIFT_PY`). For orbits, use ThriftSplat global SfM poses (`tools/sfm_poses.py`).
- MapAnything scale drifts between chunks, so stage 4 rescales the scene with the reference object.
- **Stage 3 is optional.** If objects are 3 cm apart, stage 2 separates them, and SAM only gives names.
- **Occlusion in stage 3:** z-buffer of Gaussian centres per keyframe. Do not render depth with gsplat.

---

## 5. Backends

`backends/` has one interface per model. Each interface has `local/` and `meta/` versions. The config flag `backend: local|meta` selects one.

| Interface | local (dev, free) | meta (final) |
|---|---|---|
| `segment(image, prompts)` | SAM 2.1 hiera-small (box/point prompts from stage 2 clusters) | `sam-3.1` (text prompts) |
| `describe(object_crops)` | JSON fixtures in `tests/fixtures/` | `muse-spark-1.3` vision |
| `plan(command, scene_state)` | rule parser | `muse-spark-1.3` tool calling |
| `transcribe(audio)` | not implemented | `muse-voice-transcribe-1.0` (optional, last) |

- Meta API: base URL `https://api.meta.ai/v1`, header `Authorization: Bearer $MODEL_API_KEY`. SAM 3.1 uses `POST /v1/responses`. Spark uses the OpenAI-compatible chat API.
- Prices: SAM 3.1 $2.50/1000 images, $0.20/1000 video frames. Spark 1.3 $1.25 in / $4.25 out per 1M tokens.
- The `meta` versions must return the same Python types as the `local` versions. `tests/test_backends.py` checks this with cached responses.
- **SAM 3.1 mask encoding is not documented.** The smoke test saves the raw response to `.cache/api/` so the parser can be written from it.

---

## 6. Agent tools

The agent outputs JSON tool calls. `agent/schema.py` validates each call. An invalid call returns an error message, not an exception.

- **Measure:** `measure(obj)`, `distance(a, b)`, `compare(a, b)`
- **Change physics:** `set_mass(obj, kg)`, `set_friction(obj, mu)`, `set_bounce(obj, e)`, `set_gravity(g)`
- **Act:** `push(obj, dir, force)`, `drop(obj, height)`, `stack(a, b)`, `spawn(shape, size)`, `reset()`
- **Predict:** `simulate(actions, seconds)` runs headless and returns the outcome.
- **Appearance/scene:** `set_color(obj, color)`, `scale(obj, factor)`, `remove(obj)`, `restore(obj)`, `list_objects()`.
- Requests outside the tools are reported as unsupported; the agent never runs generated code.

The simulation is deterministic: fixed timestep, no random input.

---

## 7. Folder map

```
recon/        stage 0 wrapper: calls ThriftSplat
scene_io/     load ThriftSplat output, PLY raw<->activated, camera conventions
segment/      plane, crop, cluster, sam_vote
measure/      oriented bounding box, volume, scale calibration, report
physics/      collision shapes, MuJoCo world, step loop
agent/        tool schema, parser, properties, planner
backends/     local/ and meta/ model adapters, disk cache, budget guard
viewer/       Viser app: per-object splat nodes, click/drag, command box
eval/         accuracy table vs known objects
spikes/       go/no-go scripts and paid checks
sim/          real-to-sim export, render, robot checks (branch real-to-sim)
lab/          lab bench tests (transport, grasp, gripper fit, camera) + reports
docs/         plans (REAL_TO_SIM.md)
tests/        unit tests + one small cached fixture scene
```

Each folder's `README.md` has: purpose, files (one line each), inputs, outputs, status.

---

## 8. Environment

- **Code:** `D:\metahack` (WSL: `/mnt/d/metahack`). Git runs in WSL. Line endings are LF (`.gitattributes`).
- **Data:** `LIVESPLAT_DATA=~/livesplat_data` on the WSL ext4 disk. `runs/` and `.cache/` live there. `/mnt/d` is slow for large I/O.
- **LiveSplat venv:** `~/venvs/livesplat` (Python 3.10, torch 2.6.0+cu124, SAM 2, MuJoCo, Viser).
- **ThriftSplat:** `~/projects/thriftsplat`, venv `~/projects/thriftsplat/.venv` (symlink to the MapAnything venv). Do not install LiveSplat packages into it.
- CUDA toolkit: `/usr/local/cuda-12.4` (set `CUDA_HOME`; `nvcc` is not on PATH by default).
- Viewer: open `http://localhost:8080` in the Windows browser.
- Hardware: RTX 2060 6 GB, 8 GB WSL RAM.

---

## 9. Known pitfalls (from ThriftSplat)

- Gaussian memory use is superlinear in count. Cap the count. Do not extrapolate from early iterations.
- Assign Gaussians to objects by centre, then remove Gaussians with large scale or low opacity at object edges. Large edge Gaussians leave "fuzz" on the table after an object moves.
- Objects have no visible underside. Close each object mesh with the table plane before you compute volume or the collision shape.
- Hollow objects (mug) give too much volume. Use the Spark `hollow` flag or the direct mass estimate.
- Splat PLY stores raw parameters. A viewer that gets activated values draws splats about 88x too large.
- `HF_HUB_DISABLE_XET=1` is necessary on this WSL setup. Downloads hang without it.

---

## 10. Schedule and cut order

| Date | Goal | Done when |
|---|---|---|
| Oct 28 | Repo, venv, day-1 spikes (see `STATUS.md`), capture 3 scenes, train overnight | All spikes pass or have a fallback |
| Oct 29 | Stages 1–2, objects as separate Viser nodes | Objects move as nodes |
| Oct 30 | Stage 4 + accuracy table (local) | Known objects measured |
| Oct 31 | Stage 6, MuJoCo ↔ Viser, push/drop/reset | **Core demo works** |
| Nov 1 | Agent tools, text box, `simulate`, rule parser | Text commands work |
| Nov 2 | Switch to meta backends, cache, final accuracy run | Meta path works end to end |
| Nov 3 | Buffer, debug, table hole fill, voice if time and limits allow | No blockers |
| Nov 4 | Demo video, README, submit | Submitted |

**Cut order if late:** voice → hole fill → `stack`/`spawn` → stage 3 SAM vote (keep geometry clusters only).

---

## 11. Token-saving workflow

1. Start each session: read `STATUS.md`, then the `README.md` of the folder for the task.
2. Do not read whole large files. Use Grep, then read the relevant lines.
3. Test with the small fixture scene in `tests/fixtures/`, not with a full scene.
4. Before a fix, write the hypothesis in one line. Fix one thing at a time.
5. At the end of the block: update `STATUS.md` and the changed folders' `README.md`.
