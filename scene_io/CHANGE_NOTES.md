# Changes and gaps — 2026-10-29

## Implemented
- appearance.py: evaluate existing degree 0–3 spherical-harmonic coefficients for the RGB-only viewer, using directions in original capture coordinates.
- Verified colour calculation against gsplat's reference implementation.

## Gaps / next work
- Visual comparison against original-camera renders and consistent appearance for simultaneous clients.
- This restores learned colour only; it does not add geometry, resolve occlusions or create new detail.

