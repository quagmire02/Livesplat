# backends/ — model adapters
Files:
- cache.py — Disk cache in $LIVESPLAT_DATA/.cache/api. Key = sha256 of the request.
- meta/budget.py — Ledger and hard limit (LIVESPLAT_BUDGET_USD, default 1.00).
- meta/client.py — sam31_image, spark_chat. Every call goes through the cache and the budget guard.
- meta/sam_parse.py — SAM 3.1 stream to full-frame masks (meta-sam-parser).
Rule: Paid calls only in approved API rounds (CLAUDE.md rule 1).
Network requests require LIVESPLAT_ALLOW_META=1 in addition to a key and the budget guard. Default is cached responses only; a missing entry raises PermissionRequired. Explicit cache_only stays offline even when live access is enabled. SAM retries recheck permission and budget. Muse nullable tool-message content is supported.
The user's $150 hackathon credits are not permission to call models or to raise the configured per-round budget. Keep the current budget until an approved round specifies otherwise.
Status: Total spend $0.0511.

- [CHANGE_NOTES.md](CHANGE_NOTES.md) — Short implementation summary and remaining gaps (Oct 29).
- search.py — free web + image search (DuckDuckGo via ddgs, or SearXNG). similar(): same-product links, pictures and similar products (Muse minimal picks from real results). market(): older price-listing variant. Free product search: SearXNG or DuckDuckGo (ddgs) results → Muse listings/price range/competitors, links and prices checked against the results; cached.
- baseten.py — Muse Spark Open (meta-models/muse-spark-open-1026) on Baseten credit; OpenAI-compatible; cache + budget guard; needs BASETEN_API_KEY and LIVESPLAT_ALLOW_BASETEN=1.
