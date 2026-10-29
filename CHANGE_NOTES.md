# Work summary — 2026-10-29

## Delivered
- Hosted tabletop editor with layout/history, object properties, placement and inventory.
- Contact/reflection separation and restored learned view-dependent colour.
- Visual Repair workspace with click-brush corrections and separate persistence.
- Source-reference sidebar and cached/permission-gated Muse object-card integration.
- Offline API safeguards. Latest focused checks: 52 passed, 1 skipped.

## Still to do, in practical order
1. Compare source views with renders; fix remaining boundary artifacts and view-dependent detail problems.
2. Add colour/smoothing brushes, continuous strokes and broader selection recovery.
3. Prototype small-hole repair; guide extra capture for missing surfaces. Shell-like geometry is not solved.
4. Validate Muse analysis in an approved round; add SAM refinement proposals and rendered-view diagnosis.
5. Add sourced online product/reference search. Current thumbnails are original video crops.
6. Propagate saved layouts/visual repairs to exports.
7. Medium priority: measured object dimensions and consistent metric calibration for physics/robotics.

General-scene support is deferred; focus stays on tabletops. No API calls, commits or pushes were made in these implementation passes. Live-model quality remains unvalidated.

## Folder notes
- [Viewer](viewer/CHANGE_NOTES.md)
- [Agent](agent/CHANGE_NOTES.md)
- [Backends](backends/CHANGE_NOTES.md), [Meta client](backends/meta/CHANGE_NOTES.md)
- [Reconstruction](recon/CHANGE_NOTES.md)
- [Segmentation](segment/CHANGE_NOTES.md)
- [Scene I/O](scene_io/CHANGE_NOTES.md)
- [Tests](tests/CHANGE_NOTES.md)
- [Plans](docs/CHANGE_NOTES.md)

