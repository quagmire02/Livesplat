# segment/ — stages 1–3
Files:
- plane.py — look_at_point, orbit_up, ransac_plane, find_table (densest height in cylinder + occupancy flood fill for extent).
- cluster.py — in_polygon, cluster_objects, grow_labels, attach_base_and_reflection (base band joins object; below-plane mirror image -> label -k), clean_edges.
- run.py — CLI: `python -m segment.run <run_dir>` -> align.json, labels_geo.npy, topdown.jpg.
Inputs: <run_dir>/thrift/{splat.ply, export/cameras.json}. Outputs: <run_dir>/align.json, labels_geo.npy, topdown.jpg.
Planned: sam_vote.py (optional names/split).
Status: table1: table + 3 objects correct. grow_labels removes ghost columns.
