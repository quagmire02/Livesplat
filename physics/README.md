# physics/ — MuJoCo world
Files:
- world.py — World: one free body per object (convex hull, origin = object base = viewer node frame), static table prism, floor. Actions: push (flick), slide (hand-guided, upright, stops after a distance), drop, throw, place, reset. Table/floor friction is 0.02 so each object's own friction applies (MuJoCo uses the max). Changes: mass, friction, bounce, gravity, rescale, set_removed. A rebuild keeps the state of each object by id.
- pick.py — Ray against convex hull in the object frame. Returns the nearest object.
Status: Objects settle (less than 0.04 cm). Push and stack give correct results.
