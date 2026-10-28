# viewer/ — Viser app
Files:
- app.py — Scene (background crop, object splats, reflection splats, fill splats, hulls, props.json). Compact panel: command box, log, selection, button groups. Click selects. The gizmo moves the object. Release while moving throws it. Physics runs at 60 Hz.
- holefill.py — Flat Gaussians on the object footprint, colour from the table ring, opacity feathered to the edge.
Run: `python -m viewer.app <run_dir>`, then open http://localhost:8080.
Notes: All browser clients share one panel. Do not use the disable_sliders or depth_test options of the gizmo (NaN lines).
