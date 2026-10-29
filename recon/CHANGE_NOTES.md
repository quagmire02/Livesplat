# Changes and gaps — 2026-10-29

## Implemented
- build.py: tagging is opt-in through --tag; default reconstruction does not automatically request Spark/SAM. --no-tag remains compatible.

## Gaps / next work
- Source-camera render comparison to separate rendering defects from reconstruction defects.
- Better capture guidance and registration of additional close-up views.
- Conservative hole repair/completion; no new training or completion algorithm was implemented.
- Recorded crop refinement did not improve this capture. Higher-resolution capture is a candidate, not a guaranteed fix.

