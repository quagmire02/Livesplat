"""Muse Spark Open on Baseten (hackathon credit, OpenAI-compatible). Same cache and budget guard as the Meta client.

Key: BASETEN_API_KEY (~/.livesplat_env). Model name must be exactly meta-models/muse-spark-open-1026.
Spark Open always reasons; the answer is in message.content (reasoning, if returned, is kept separately).
"""
import os

import requests

from backends import cache
from backends.meta import budget
from backends.meta.client import PermissionRequired

BASE = "https://inference.baseten.co/v1"
MODEL = "meta-models/muse-spark-open-1026"
TIMEOUT = 180


def chat(messages, tools=None, max_tokens=2000, *, cache_only=False):
    req = {"model": MODEL, "messages": messages, "max_tokens": max_tokens}
    if tools:
        req["tools"] = tools
    hit = cache.get(req)
    if hit is not None:
        return hit
    if cache_only:
        raise PermissionRequired("This Baseten request is not cached.")
    key = os.environ.get("BASETEN_API_KEY")
    if not key:
        raise RuntimeError("BASETEN_API_KEY is not set")
    if os.environ.get("LIVESPLAT_ALLOW_BASETEN") != "1":
        raise PermissionRequired("Baseten calls are disabled; set LIVESPLAT_ALLOW_BASETEN=1 for an approved round.")
    n_img = sum(1 for m in messages if isinstance(m.get("content"), list)
                for c in m["content"] if c.get("type") == "image_url")
    chars = sum(len(str(m.get("content") or "")) for m in messages if not isinstance(m.get("content"), list)) + len(str(tools or ""))
    est = budget.spark_estimate(chars, n_img, max_tokens)     # conservative: Spark 1.3 prices
    budget.check(est, "baseten spark-open")
    usd = est
    try:
        r = requests.post(f"{BASE}/chat/completions", json=req, timeout=TIMEOUT,
                          headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            out = r.json()
        except ValueError:
            out = {"error": r.text[:2000]}
        out["_status"] = r.status_code
        u = out.get("usage") or {}
        if "prompt_tokens" in u:
            usd = (u["prompt_tokens"] * budget.PRICE["muse-spark-1.3/in"]
                   + u.get("completion_tokens", 0) * budget.PRICE["muse-spark-1.3/out"])
    finally:
        budget.record(usd, "baseten spark-open (Baseten credit)")
    if out.get("_status") == 200:
        cache.put(req, out, {"model": MODEL})
    return out
