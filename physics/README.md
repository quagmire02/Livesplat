# physics/ — stage 6
Files:
- pick.py — ray vs convex hull (slab test) in each object frame; nearest hit.
- world.py — World: MuJoCo model from objects (convex-hull mesh per free body, origin = object base = viewer node frame), static table prism from polygon, floor plane. Actions: push, drop, throw, reset, set_mass/friction/bounce (rebuild model, keep state), set_gravity. DT = 2 ms.
Inputs: objects [{id, base, hull, mass, friction, bounce}], polygon, floor_z. Built by viewer.app.Scene.
Status: table1 settles (<0.04 cm drift); push tips the spray can over as expected.
