"""Robust JSON from model replies, shared by every Muse feature.

Long structured replies occasionally break JSON (seen: a stray token mid-object, '"at": "top"} widespread: null}';
also code fences, trailing commas, smart quotes). loads() tries: the reply as is -> local clean-ups -> one cheap
repair call to Muse ("return the same content as valid JSON") -> None. The repair call is cached like any other.
"""
import json
import re


def _first_object(t):
    i, j = t.find("{"), t.rfind("}")
    return t[i:j + 1] if i >= 0 and j > i else None


def _local_fixes(t):
    t = t.replace("“", '"').replace("”", '"').replace("’", "'")
    t = re.sub(r"```(?:json)?", "", t)
    t = re.sub(r",\s*([}\]])", r"\1", t)                       # trailing commas
    return t


def try_parse(text):
    """Parse without any model call (None if it fails)."""
    if not text:
        return None
    for t in (text, _local_fixes(text)):
        obj = _first_object(t)
        if obj is None:
            continue
        try:
            return json.loads(obj)
        except ValueError:
            continue
    return None


def loads(text, repair=True, cache_only=False):
    """Parse a model reply; if broken, ask Muse once to repair it. Returns a dict/list or None."""
    j = try_parse(text)
    if j is not None or not repair or not text or "{" not in text:
        return j
    try:
        from backends.meta import client
        r = client.spark_chat([{"role": "user", "content": "This JSON is invalid. Return the same content as valid "
                                                           "JSON only, changing nothing else:\n" + text[:8000]}],
                              max_tokens=4000, cache_only=cache_only, extra={"reasoning_effort": "minimal"})
        return try_parse((r.get("choices") or [{}])[0].get("message", {}).get("content") or "")
    except Exception:
        return None


def reply_json(r, repair=True, cache_only=False):
    """loads() on a chat-completion response dict."""
    c = ((r or {}).get("choices") or [{}])[0].get("message", {}).get("content") or ""
    return loads(c, repair, cache_only)
