# segment/ — table and objects (stages 1–2)
Files:
- plane.py — find_table: up = camera-orbit normal; table height = densest height in a cylinder; table extent = occupancy flood fill.
- cluster.py — cluster_objects (DBSCAN above the table), grow_labels (adds faint Gaussians near an object), attach_base_and_reflection (base joins the object; mirror image below the table gets label -k), clean_edges.
- run.py — `python -m segment.run <run_dir> [--src thrift_sfm]`. Writes align.json, labels_geo.npy, topdown.jpg.
- sam_vote.py — SAM 3.1 on every 3rd keyframe x one prompt per object (props.json sam_prompt); instance matched by overlap with geometric labels; z-buffer visibility; SAM may add Gaussians (>= 50%) and move them between objects (>= 60%, current < 20%), never to background (masks miss object parts). Writes labels.npy, sam_report.json. Paid (~$0.19 for table1).
Limit: Touching objects become one object. SAM split is not done.
Status: table1 gives the table and 3 correct objects.
