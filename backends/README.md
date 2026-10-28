# backends/ — model adapters
Files:
- cache.py — disk cache in $LIVESPLAT_DATA/.cache/api, key = sha256(request).
- meta/budget.py — ledger + hard cap (LIVESPLAT_BUDGET_USD, default 1.00). Prices from dev.meta.ai.
- meta/client.py — sam31_image (raw stream kept), spark_chat (OpenAI-compatible). Cache + budget on every call.
- local/ — planned: sam2 segment, fixture describe, rule plan.
Status: meta client written, not yet called.
