# Changes and gaps — 2026-10-29

## Implemented
- parser.py/schema.py: distinguish object IDs from quantities and validate malformed/numeric tool arguments.
- controller.py: isolated prediction state, yaw-versus-tilt correction, safer scaling and stationary captured reflections.
- planner.py: local/cache-only paths, nullable Muse responses and selected-object grounding.

## Gaps / next work
- Approved live Muse evaluation on real commands and object-analysis results.
- Preserve measured versus estimated property provenance as calibration is added.
- Model repair suggestions with previews; do not silently apply uncertain physical properties.

