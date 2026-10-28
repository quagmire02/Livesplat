# viewer/ — Viser app
Files:
- app.py — Scene (table frame, bg crop, per-object splat + reflection + hole-fill splats, outlier-filtered hull with flat footprint, volume, default mass, floor height). main(): physics loop 60 Hz; command box (planner local|meta), log; object dropdown; actions/world folders. `python -m viewer.app <run_dir> [--src thrift_sfm]`.
- holefill.py — fill_splat: flat Gaussian grid on the object footprint, colour = IDW of table ring (brightness outliers dropped). Shown while the object is away from rest.
Node rules: reflection follows its object (xy + yaw) while upright on the table; hidden when lifted/tipped/off. Fill visible when not at rest.
Notes: GUI state is shared by all browser clients (two people typing collide).
Status: commands, recolour, stack, hole fill verified on table1.
