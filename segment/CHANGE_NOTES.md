# Changes and gaps — 2026-10-29

## Implemented
- complete3d.py: SAM masks are strictly cache-only, preventing unexpected paid requests during local collision-mesh generation.
- Documented TripoSR's collision-only role; visual contact separation lives in viewer/contact.py, not rewritten labels.

## Gaps / next work
- Touching-object separation, multi-view boundary validation and approved SAM refinement proposals.
- Current object-card crops reuse existing segmentation; they are not fresh SAM masks.
- Table/object ownership remains heuristic; viewer repairs do not yet update segmentation/export labels.

