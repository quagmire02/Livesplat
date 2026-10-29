# Changes and gaps — 2026-10-29

## Implemented
- client.py requires LIVESPLAT_ALLOW_META=1 before live requests; explicit cache_only never falls through to network.
- SAM retries recheck permission/budget; Spark handles nullable message content.
- New object-card analysis reuses spark_chat, caching and the existing budget guard.

## Gaps / next work
- Live object-card analysis has not been exercised in this work block; offline tests/cache-miss checks only.
- Online search capability and source attribution need a separate verified integration.
- User approval is still required before an API round; no budget increase was made.

