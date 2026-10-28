# backends/ — model adapters
Files:
- cache.py — disk cache in $LIVESPLAT_DATA/.cache/api, key = sha256(request).
- meta/budget.py — ledger + hard cap (LIVESPLAT_BUDGET_USD, default 1.00). Prices from dev.meta.ai.
- meta/client.py — sam31_image (raw SSE kept), spark_chat (OpenAI-compatible). Cache + budget on every call.
- meta/sam_parse.py — parse_image(raw, (H,W)) -> full-frame masks via meta-sam-parser (official decoder).
- local/ — planned: sam2 segment, fixture describe, rule plan.
Status: smoke test PASS ($0.0176 total). Spark is a reasoning model: use max_tokens >= 1500.
