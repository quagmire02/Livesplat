# Changes and gaps — 2026-10-29

## Implemented
- app.py, editor.py, session.py: hosted inspector, precise placement, properties, inventory CSV, layout save/load and undo/redo; physics starts paused.
- contact.py, holefill.py: separate captured contact/reflection from moving objects; fill exposed table using the scan footprint. Original scan files stay intact.
- repair.py: isolate, click-brush preview, erase/restore, object/contact reassignment, repair undo and separate save/load.
- object_card.py: left source-frame thumbnails, cached/permission-gated Muse analysis, uncertainty and capture suggestions, explicit undoable name confirmation.
- app.py restores learned view-dependent colours; holefill_ai.py now checks permission before a cache-miss request.

## Gaps / next work
- Compare rendered views with original frames; shell-like surfaces and missing backsides remain unresolved.
- Colour/smoothing brushes, continuous strokes, background-to-object recovery, small-hole repair and guided recapture.
- Fresh SAM masks, rendered-view Muse diagnosis, approved live analysis validation and sourced online product thumbnails. Current thumbnails are source crops, not product matches.
- Repair/layout export into simulation; repair overlays currently affect this viewer only.
- Medium priority: measured dimensions and metric calibration. Current dimensions are scene units.
- Multi-client shading uses the most recently updated camera; brush depth selection is heuristic.


## 2026-10-29 (later): SAM-grounded object reference panel
- sam_isolate.py: Isolate with SAM 3.1 in 3 distinct source views (cache-first: same frames/prompts as segment.sam_vote, $0 on tagged scenes). Per view: coverage (mask filled by the reconstruction) and spill (object splats outside the mask). Masked thumbnails + diagnosis overlays (green outline, magenta = not reconstructed, red = attachments).
- object_card.py: "Isolate with SAM 3.1" button; Muse analysis uses masked crops + overlays + metrics (PROMPT_SAM) and returns category, scan_issues, capture_next, repair_advice (remove_attachments|recapture|none). Suggested fix = splats outside the masks in 2+ views -> "preview fix" (yellow, Repair workspace) / "apply fix" (keep on table; undo/save in Repair).
- table1 (cached, $0): coverage 98-100% in all views; suggested fixes bottle 6,268, can 525, roll 3,653 splats (table skirt at the base, fuzz).
- Not done: live Muse validation of PROMPT_SAM (needs an approved round); online product search (needs a search source + permission). Reference photos stay evidence only, never geometry.
