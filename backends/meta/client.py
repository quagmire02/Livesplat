"""Thin Meta Model API client. Every call goes through the cache and the budget guard.

Base URL https://api.meta.ai/v1, key in MODEL_API_KEY.
SAM 3.1: POST /responses (streamed, raw text kept because the mask encoding is undocumented).
Spark:   POST /chat/completions (OpenAI-compatible).
"""
import base64
import io
import os

import requests

from backends import cache
from backends.meta import budget

BASE = "https://api.meta.ai/v1"
TIMEOUT = 120


class PermissionRequired(RuntimeError):
    """A cache miss needs an explicitly approved API round."""


def require_permission():
    if os.environ.get("LIVESPLAT_ALLOW_META") != "1":
        raise PermissionRequired("No cached response. API calls are disabled; approve a round before enabling Meta.")


def _headers(stream=False):
    k = os.environ.get("MODEL_API_KEY")
    if not k:
        raise RuntimeError("MODEL_API_KEY is not set")
    h = {"Authorization": f"Bearer {k}", "Content-Type": "application/json"}
    if stream:
        h["Accept"] = "text/event-stream"
    return h


def image_data_uri(pil_image, fmt="JPEG", quality=90) -> str:
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, fmt, quality=quality)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def sam31_image(prompt: str, image_uri: str, *, cache_only=False) -> dict:
    """One SAM 3.1 image call. Returns {'status', 'raw'} with the raw stream text."""
    req = {"model": "sam-3.1", "stream": True, "input": [{
        "type": "message", "role": "user", "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": image_uri}]}]}
    hit = cache.get(req)
    if hit is not None:
        return hit
    if cache_only:
        raise PermissionRequired("SAM response is not cached; this operation is cache-only.")
    est = budget.PRICE["sam-3.1/image"]
    for attempt in range(2):
        require_permission()
        budget.check(est, "sam-3.1 image")
        try:
            r = requests.post(f"{BASE}/responses", headers=_headers(stream=True), json=req,
                              stream=True, timeout=TIMEOUT)
            out = {"status": r.status_code, "raw": r.text}
            break
        except requests.exceptions.Timeout:
            if attempt:
                raise
        finally:
            budget.record(est, "sam-3.1 image", {"prompt": prompt, "attempt": attempt + 1})
    if out["status"] == 200:
        cache.put(req, out, {"model": "sam-3.1", "prompt": prompt})
    return out


def spark_chat(messages: list, tools: list | None = None, max_tokens: int = 300,
               model: str = "muse-spark-1.3", *, cache_only=False, extra: dict | None = None) -> dict:
    """One Spark chat call. Returns the parsed JSON response (or error dict).
    extra: additional request fields (e.g. reasoning_effort); omitted from the request when None."""
    req = {"model": model, "messages": messages, "max_tokens": max_tokens, **(extra or {})}
    if tools:
        req["tools"] = tools
    hit = cache.get(req)
    if hit is not None:
        return hit
    if cache_only:
        raise PermissionRequired("This Muse request is not cached. Choose local commands or approve a live API round.")
    require_permission()
    n_img = sum(1 for m in messages if isinstance(m.get("content"), list)
                for c in m["content"] if c.get("type") == "image_url")
    chars = sum(len(c.get("text", "")) if isinstance(c, dict) else len(c)
                for m in messages for c in (m.get("content") if isinstance(m.get("content"), list)
                                           else [m.get("content") or ""]))
    chars += sum(len(str(m.get("tool_calls") or "")) for m in messages)
    chars += len(str(tools or ""))
    est = budget.spark_estimate(chars, n_img, max_tokens)
    budget.check(est, f"{model} chat")
    usd = est
    try:
        r = requests.post(f"{BASE}/chat/completions", headers=_headers(), json=req, timeout=TIMEOUT)
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
        budget.record(usd, f"{model} chat")
    if out.get("_status") == 200:
        cache.put(req, out, {"model": model})
    return out
