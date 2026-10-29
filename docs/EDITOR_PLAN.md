# Editor feature plan

## Product and constraints

LiveSplat serves three connected uses: ordinary users explore/rearrange their scanned objects;
measurements help everyone; labs run physics experiments and optional robot checks.
The shared artifact is one scene with object IDs, geometry, names, properties and saved layouts.

Meta/Muse is the enhanced path for the Meta AI Global Hackathon. The user has $150 of credit,
but that is not permission to spend it. Finish the baseline first; use approved rounds for tuning
and the final demo. Local commands and cached responses must remain usable afterwards.
No API calls, commits or pushes in this work block. Branch: codex/new-features.

## What the finished first pass looks like

The existing Viser web app has four tabs:

| Tab | User-facing result |
| --- | --- |
| Explore | Click/drag objects, choose Arrange or Experiment, optional throw-on-release, local/cached/live-approved commands |
| Inspect | Object list, dimensions with honest units, names and editable properties, precise upright placement, absolute size, restore/remove, separation comparison, undo/redo, save/load, CSV inventory, camera presets |
| Clean up | Existing erase, protect-table and undo controls |
| Robotics | Existing arm selection, base placement and grasp checks |

Arrange pauses physics so a layout stays where the user places it. Experiment runs physics.
Explicit action commands (push/drop/throw/stack) resume physics. Undo/load pause it so results
can be inspected. Saved layouts are separate from reconstruction and reject incompatible geometry.
Object additions form a history boundary; background cleanup keeps its separate undo history.

## Implementation sequence and acceptance

1. **Command / offline foundations:** fix quantities vs IDs, nullable Muse content, malformed tool
   arguments, yaw-vs-tilt and prediction isolation. Disable network cache misses by default.
   Acceptance: small synthetic tests and mocked model conversations; zero external requests.
2. **Editor session:** compact undo/redo and atomic layout save/load, geometry identity check,
   names and CSV inventory. Acceptance: edit, undo, redo, save, reset, load restores the same layout.
3. **Hosted controls:** connect all first-pass features to Viser tabs. Acceptance: the server starts
   on the local web host and the actual GUI callbacks perform the intended changes, including
   while physics is paused. Do not count a CLI-only helper as a completed UI feature.
4. **Hand-off:** update folder indexes and STATUS, leave changes uncommitted, report the running
   address and any remaining visual-verification limitations.

Needed: existing WSL LiveSplat venv, local table1 capture, existing Viser/MuJoCo libraries.
No new dependencies or model downloads are needed for this pass.

## Medium priority: user-entered dimensions and calibration

Add a Measure / Calibration section in Inspect, also accessible from Robotics:

- Select an object, enter a known width/depth/height and its units, and record the source.
- **Reference calibration:** a measured dimension determines one global metres-per-scene-unit
  factor. Apply it consistently to all geometry, cameras, motion quantities and robot exports.
- **Object correction:** adjust an individual object's dimensions. Make it explicit that this
  changes the reconstructed object and may disagree with its original appearance.
- Show measured vs estimated vs uncalibrated state; never silently treat typical product sizes
  from a model as measured scale. Report disagreement between multiple entered references.
- Preserve mass supplied by a user; recompute density-derived mass when appropriate. Inertia,
  collision geometry and gripper fit must follow corrected dimensions.
- Save calibration/provenance with the scene and export it. Test one known-size synthetic object
  and one manually measured capture end to end before presenting centimetres as reliable.

This is deferred, not included in the first-pass controls. Dimensions and distances currently
remain explicitly in scene units. Robot results remain approximate without calibration.

## Data already available and useful next features

| Available evidence | Useful result | Limitation |
| --- | --- | --- |
| Gaussian geometry, labels and object hulls | Inventory, object dimensions, positions, separation, selection and layout | Missing surfaces; hulls are approximations |
| Capture images and calibrated camera poses | Show original object crops, choose additional views, explain recognition | Occlusion and reflections can corrupt views |
| Object poses and table boundary | Layout comparison, tabletop occupancy, approximate fit checks | Need metric calibration and proper surface clearance |
| MuJoCo state and collision shapes | What-if previews, repeatable experiments, reset and replay | Mass/friction/shape uncertainty affects results |
| Names, aliases and property estimates | Natural-language selection and commands, editable object cards | Model output is evidence to review, not ground truth |
| Robot models and exported scenes | Reach/grasp comparisons and experiment reports | Sim success alone does not establish real-world success |

## Muse / Meta precision work after the baseline

These are proposed uses of the existing vision, segmentation and tool interfaces; no new API
capabilities or performance gains are assumed.

1. **Multiple views per object:** send a few sharp, different views instead of one crop, including
   scene context and object IDs. Ask for names/materials and visible evidence; allow unknown.
   Compare with user-corrected labels. This can help distinguish a full roll from an empty tube.
2. **Segmentation repair:** use model masks as proposals, then check multi-view consistency,
   depth visibility and connected geometry. Preview the change before accepting it. Add manual
   split/merge so touching objects can be recovered when model masks are imperfect.
3. **Grounded scene questions:** give Muse structured geometry and measured properties, let tools
   compute distances/fit/physics, and have Muse explain those results. Do not ask it to invent dimensions.
4. **Better command grounding:** include current selection, object names, removed state, geometry,
   action limits and units. Support read-tool -> action-tool follow-ups with bounded rounds.
5. **Uncertainty-aware experiments:** expose mass/friction ranges and compare outcomes across a
   few values. Vision cannot determine contents, mass or friction exactly from appearance alone.

Before an approved round: fix the test scene and prompts, list the intended calls and budget,
reuse existing cache entries, then compare results with the baseline. Keep a separate recording
of live Muse use for the demo; cached replay is labelled cached, never represented as a live call.

## Deferred

Manual calibrated dimensions (medium priority), visual trajectory preview, scene export of edited
layouts, source-crop cards, segmentation split/merge and multi-view Muse enrichment follow this pass.
Voice, additional robots and a Gym interface are lower priority than these core workflows.
