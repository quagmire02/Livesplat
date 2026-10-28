"""Budget guard for paid Meta API calls.

Every paid call is written to ledger.jsonl with its cost. A call is refused
if spent + worst-case estimate would exceed LIVESPLAT_BUDGET_USD (default 1.00).
Failed calls are still charged at their estimate (pessimistic).
"""
import json
import os
import time

from backends.cache import cache_dir

PRICE = {
    "sam-3.1/image": 2.50 / 1000,          # per image
    "sam-3.1/frame": 0.20 / 1000,          # per video frame
    "muse-spark-1.3/in": 1.25 / 1e6,       # per input token
    "muse-spark-1.3/out": 4.25 / 1e6,      # per output token
}
IMAGE_TOKENS_WORST = 3000                  # worst-case tokens per input image (not documented)


class BudgetExceeded(RuntimeError):
    pass


def ledger_path():
    return cache_dir() / "ledger.jsonl"


def budget() -> float:
    return float(os.environ.get("LIVESPLAT_BUDGET_USD", "1.00"))


def spent() -> float:
    p = ledger_path()
    if not p.exists():
        return 0.0
    return sum(json.loads(l)["usd"] for l in p.read_text().splitlines() if l.strip())


def check(estimate_usd: float, what: str):
    s = spent()
    if s + estimate_usd > budget():
        raise BudgetExceeded(f"{what}: spent ${s:.4f} + est ${estimate_usd:.4f} > budget ${budget():.2f}")


def record(usd: float, what: str, extra: dict | None = None):
    with open(ledger_path(), "a") as f:
        f.write(json.dumps({"t": time.time(), "what": what, "usd": usd, **(extra or {})}) + "\n")


def spark_estimate(text_chars: int, n_images: int, max_tokens: int) -> float:
    tin = text_chars / 3 + n_images * IMAGE_TOKENS_WORST
    return tin * PRICE["muse-spark-1.3/in"] + max_tokens * PRICE["muse-spark-1.3/out"]
