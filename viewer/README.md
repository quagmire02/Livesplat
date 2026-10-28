# viewer/ — Viser app
Files:
- app.py — UI: dark theme, title bar, tabs Play / Clean up / Simulate, one-click example commands from object names. Scene (collision hull from <run>/gen/obj_k.obj with a flat base if it stands alone (physics.world.stands_alone), else the scan hull; background crop, object splats, reflection splats, fill splats, hulls, props.json). Compact panel: command box, log, selection, button groups. Click selects. The gizmo moves the object. Release while moving throws it. Physics runs at 60 Hz.
- holefill.py — Flat Gaussians on the object footprint, colour from the table ring, opacity feathered to the edge.
- holefill_ai.py — Muse Image fill: top-down gsplat render, footprints painted magenta (no mask API), edit "clean wood, no reflections"; grey pixels -> table wood colour, sampled into gen/fill_rgb_k.npy (Scene uses it). `sim/run.sh viewer.holefill_ai <run>`. ~$0.01.
- add_object.py — Click to add a missed object: first DENSE surface along the ray (not a floater) -> SAM 3.1 point prompts on up to 6 visible keyframes -> multi-view vote (>= 3 views) over nearby non-table splats, connected piece at the click. Viewer button "add missed object". ~$0.015 per click. Persists in labels.npy.
Run: `python -m viewer.app <run_dir>`, then open http://localhost:8080.
Clean-up: folder "clean up background" (tabletop mode): Shift + drag a box erases background splats (projection from the client camera), "protect table" (whole table column: glossy tops are faint + reflection splats below the surface), auto clean (radius; min opacity default 0), undo. Saved to <run>/bg_removed.npy (raw indices); used by Scene and sim.export.
Notes: All browser clients share one panel. Do not use the disable_sliders or depth_test options of the gizmo (NaN lines).
