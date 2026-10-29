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

