# backends/ — model adapters
Files:
- cache.py — Disk cache in $LIVESPLAT_DATA/.cache/api. Key = sha256 of the request.
- meta/budget.py — Ledger and hard limit (LIVESPLAT_BUDGET_USD, default 1.00).
- meta/client.py — sam31_image, spark_chat. Every call goes through the cache and the budget guard.
- meta/sam_parse.py — SAM 3.1 stream to full-frame masks (meta-sam-parser).
Rule: Paid calls only in approved API rounds (CLAUDE.md rule 1).
Status: Total spend $0.0511.
