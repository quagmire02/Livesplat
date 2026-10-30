# LiveSplat: connected Lab and Guide plan

Planning snapshot: 2026-10-30. Based on the user's updated overview and current project notes/code indexes. Reported results are not independently revalidated by this planning pass. This document supersedes the old editor-only roadmap for these two extensions.

## Product promise

**Scan a workspace. Learn what to do with it, or test what will work before acting.**

One scene, two explicit experiences:
- **Guide:** explain, suggest and teach using the user's actual objects.
- **Lab:** test equipment/actions, identify decision-changing uncertainty, collect evidence and compare interventions.

Keep the existing Ask/Tell entry point. Route clear requests automatically and show the chosen mode; ask only when intent is ambiguous. Do not add another collection of unrelated tabs. General-scene support remains deferred.

## 1. Lab: from result cards to evidence-driven decisions

### Already present
Transport, gripper fit, arm grasp and camera tests; parameter sweeps; reports/replays; typed dimension calibration; advisor sensitivity summaries and simulated proposed changes. Reuse these, rather than rebuilding their capabilities.

### New user journey
1. **Define a goal and criterion.** Example: lift this bottle 10 cm and hold for 2 seconds with this arm. Confirm reachable equipment and test limits. Keep transport, gripper and camera criteria task-specific.
2. **Establish a reproducible baseline.** Record scene/layout revision, geometry source, equipment spec, calibration, parameter ranges, trial settings and seeds. Summarize pass count and failure types.
3. **Explain the deciding uncertainty.** Extend existing advisor outputs with the underlying conditional results. Distinguish slip, collision, reach, width and uncertain geometry. Include interactions; a one-variable sensitivity ranking is a heuristic, not causal proof.
4. **Request one useful observation.** Choose among supported actions: weigh object, measure grip width/height, provide a clearer side view, verify gripper specifications, or record a controlled transport observation. Rank by decision sensitivity and user effort. If uncertainty cannot be resolved this way, say so.
5. **Update evidence, not just a slider.** Store value/range, units, method, source and timestamp. A width measurement constrains scale/width, not friction or mass. Preserve original assumptions and do not silently narrow unrelated ranges.
6. **Rerun and compare.** Use matched scenarios/seeds for baseline and intervention comparisons. Show what changed because of new knowledge separately from what changed because of a physical modification.
7. **Choose an intervention.** Reuse advisor proposals: pads, grip position, force within equipment limits, base placement or carrier settings. Simulate supported changes; never present a model suggestion as a measured improvement.
8. **Record a real observation.** User records measured outcome or attaches local evidence. Show prediction versus observation and remaining disagreement. No autonomous hardware control.

### What the screen contains
One decision card: goal; verdict under tested assumptions; failure reason; next measurement; before/after comparison; watch trial; export report. Expandable evidence drawer contains assumptions and provenance. Annotate contact locations, collision regions, camera blind spots or transport direction directly on the scene.

### First vertical slice
Transport: predicted slide/tip threshold -> user-entered controlled observation -> evidence update -> rerun -> held-out physical check. For quasi-static sliding, a tilt observation can constrain an effective friction range only if the object slides before tipping; a tipping observation must not be treated as a friction measurement. Record procedure and uncertainty. This is a proposed experiment, not an existing validated estimator.

Then reuse the same evidence workflow for gripper fit and grasp. Camera results can request additional capture when the relevant surface is poorly observed.

### Acceptance
- One new measurement updates only relevant assumptions; old and new reports remain reproducible.
- Each verdict links to actual trials; stale results are marked when scene/equipment changes.
- Show a simulated intervention and, where available, an independent physical check.
- Trial pass fractions are robustness across chosen scenarios, not real-world probabilities. Cross-object scale spread is not measured accuracy. Real force limits distinguish sourced specifications from estimates.

## 2. Guide: teach with the scanned objects

### Already present
Explain grounds parts with SAM and adds narration; Guide highlights objects step by step; routing, source photos, Similar search and completed visual meshes exist. Extend them with a structured storyboard and drawing player.

### Two entry points
- **How do I…?** Turn a specified goal into a short visual lesson.
- **What can I do with these?** Suggest up to three grounded activities with required objects, missing items and rough effort. Let the user choose before generating a full lesson. Avoid pretending that a partial scan inventories the entire workspace.

### Lesson flow
1. Resolve real object IDs; confirm uncertain identities and missing prerequisites.
2. Generate a validated storyboard: learning goal, steps, objects, camera framing, highlights, paths, illustration brief, narration, completion prompt and dependencies.
3. Present one step at a time with Next/Back/Repeat/Why. Camera frames the relevant objects; an optional presentation copy brings an object forward without altering its physical scene pose.
4. Animate screen-space illustrations and 3D-anchored arrows/outlines. Labels use measured or observed facts where available.
5. User confirms completion. The scanned scene does not sense real progress; no automatic claim that an egg was cracked or a stove switched off.
6. Clarification revises the current step or remaining lesson, retaining completed steps and cached assets.

### Model and renderer responsibilities
- **Muse Spark:** grounded lesson plan, constrained drawing commands, clarification and narration text.
- **SAM:** reuse existing object/part masks; request new masks only when needed and approved.
- **Muse Image:** custom reference-conditioned illustrations of actions or transformations; generate per step or reusable asset, not per animation frame.
- **Browser:** validate commands and animate outlines, arrows, labels, paths, timers and image reveals; no arbitrary generated JavaScript.
- **Existing speech path:** speak approved step text; align player timing to actual audio where available. Do not assume image generation supplies speech.

First lesson: one coherent egg-sandwich walkthrough using a suitable new capture. Eggs cracking, cooking and spreading are explicitly illustrative overlays; no soft-body or cooking simulation claim. Check missing utensils/ingredients before teaching. Review generated action depictions before caching a demo-ready lesson.

### Acceptance
- Every highlight targets an existing object/part ID or an explicitly labelled missing item.
- Steps work forward/backward and preserve the original scene layout.
- Illustrations are distinct from measured scans and simulation replays.
- A cached lesson plays without new model calls; failed generation falls back to text and simple drawings.
- Demonstrate both a how-to request and a what-can-I-do suggestion leading into the same player.

## 3. The shared connection

```text
Scan + object IDs + source views + properties/provenance
                         |
                   Ask/Tell router
                 /                 \
       Guide storyboard       Lab experiment plan
       and illustrations      and measured trial results
                 \                 /
            Shared visual step/replay player
       camera framing, highlights, arrows, narration
                         |
           User confirmation / new evidence
                         |
         Update lesson or rerun affected trials
```

Shared records:
- **Scene revision:** layout, object IDs, geometry/appearance versions and calibrated units.
- **Evidence:** observed, user-measured, model-estimated, simulated or generated; source and uncertainty.
- **Step:** object references, bounded visual commands, narration, asset references and advance condition.
- **Experiment:** goal, equipment, assumptions, trial configuration, outcomes and baseline link.

A lab result can open a visual explanation using Guide's player. A general user can ask whether a tray arrangement is stable and enter a Lab test. Guide must not route cooking actions into rigid-body physics. Generated illustrations must never become calibration evidence or collision truth.

## 4. Implementation order and boundaries

1. **Shared foundation:** scene/evidence revision IDs, storyboard schema and player, stable object/part anchors. Extend viewer/route.py, viewer/explain.py and the existing app integration; keep UI work isolated from the frozen consumer baseline until ready.
2. **Lab depth:** extend lab/advisor.py and viewer/lab_panel.py with next-observation actions, evidence entry and matched before/after runs. Reuse measure/scale.py; do not rebuild calibration. Deliver transport first, then grasp.
3. **Guide visuals:** connect the player to existing Guide/Explain, add how-to/what-to entry paths, then Muse Image assets and cache replay. Keep rendering independent of live model availability.
4. **Demo integration:** one short consumer lesson, one Lab decision/evidence/intervention loop; export results and capture cached playback. Test grounding, state restoration, cache misses and one full happy path for each mode rather than many speculative edge cases.

Needed: current local scan + reports, one appropriate teaching capture, one feasible physical validation setup, user measurements, confirmed equipment limits, current model credentials and an explicitly approved call budget. No new robot hardware is required for the first transport demonstration.

## 5. Credit and demo strategy

Approximately $140 remains according to the user; that is not spending authorization. Proposed first approved round: total cap $5, with at most $2 for image experiments and the remainder for lesson planning/clarification and Lab advice. Keep a reserve for final tuning. Confirm current pricing before running; enforce the approved cap across all calls and retries. No calls are authorized by this plan itself.

Cache by scene/evidence revision + request + model/prompt version. Asset completion must not switch a user's current step unexpectedly. Precompute demo assets; label cached playback honestly. Track request count, latency and spend. A limited live clarification can demonstrate responsiveness if explicitly approved and rehearsed.

## Demo narrative

**Consumer:** "What can I make with these?" -> choose sandwich -> real objects highlighted -> custom illustrated step -> ask why -> continue.

**Lab:** "Can this setup carry/handle this object?" -> measured simulated failure -> request one observation -> update evidence -> test an intervention -> compare with a physical observation.

The claim is a grounded, evidence-aware workspace assistant. Do not claim novel reconstruction, universal teaching, certified robotics, or solved unseen geometry.
