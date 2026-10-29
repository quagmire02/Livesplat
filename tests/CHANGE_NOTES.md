# Changes and gaps — 2026-10-29

## Implemented
- test_editor.py: layout/history, inventory, prediction isolation and command validation.
- test_backend_offline.py: mocked model responses, permission gates and cache-only behaviour.
- test_splat_visuals.py: contact separation and SH reference comparison.
- test_repair.py: brush front-depth selection, history and repair-file validation.
- test_object_card.py: structured analysis validation and distinct camera crops.
- Latest focused run: 52 passed, 1 skipped. Browser checked placement, repair erase/undo, reference thumbnails and safe cache miss.

## Gaps / next work
- Approved real-model evaluation; currently no live analysis quality claim.
- Original-view visual comparisons and targeted real-scene boundary/hole checks.
- Calibration/export tests when those features are implemented. Software-WebGL browser checks are not GPU performance benchmarks.

