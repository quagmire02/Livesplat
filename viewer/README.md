# viewer/ — Viser app
Files:
- app.py — Scene (bg crop, per-object splat + reflection + fill, outlier-filtered hull with flat footprint, mass, floor). main(): compact floating panel (command box, log, selection, button groups); click-to-select (physics/pick.py ray vs hull); transform gizmo at mid-height drags the object kinematically, release while moving = throw (>0.5 m/s, 60%, cap 2.5); 60 Hz physics loop.
- holefill.py — fill_splat: flat Gaussian grid on the object footprint, colour = IDW of table ring (brightness outliers dropped). Shown while the object is away from rest.
Node rules: reflection follows its object (xy + yaw) while upright on the table; hidden when lifted/tipped/off. Fill visible when not at rest.
Notes: GUI state is shared by all browser clients. Gizmo with disable_sliders/depth_test options rendered NaN lines; use defaults.
Status: commands, recolour, stack, hole fill verified on table1.
