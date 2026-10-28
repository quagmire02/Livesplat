# viewer/ — Viser app
Files:
- app.py — Scene (collision hull from <run>/gen/obj_k.obj with a flat base if it stands alone (physics.world.stands_alone), else the scan hull; background crop, object splats, reflection splats, fill splats, hulls, props.json). Compact panel: command box, log, selection, button groups. Click selects. The gizmo moves the object. Release while moving throws it. Physics runs at 60 Hz.
- holefill.py — Flat Gaussians on the object footprint, colour from the table ring, opacity feathered to the edge.
Run: `python -m viewer.app <run_dir>`, then open http://localhost:8080.
Clean-up: folder "clean up background" (tabletop mode): Shift + drag a box erases background splats (projection from the client camera), "protect table top", auto clean (keep within N x table radius, min opacity), undo. Saved to <run>/bg_removed.npy (raw indices); used by Scene and sim.export.
Notes: All browser clients share one panel. Do not use the disable_sliders or depth_test options of the gizmo (NaN lines).
