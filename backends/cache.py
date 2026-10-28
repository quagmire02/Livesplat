"""Disk cache for model calls. Key = sha256 of the canonical JSON request."""
import hashlib
import json
import os
from pathlib import Path


def data_dir() -> Path:
    return Path(os.environ.get("LIVESPLAT_DATA", Path.home() / "livesplat_data"))


def cache_dir() -> Path:
    d = data_dir() / ".cache" / "api"
    d.mkdir(parents=True, exist_ok=True)
    return d


def key(request: dict) -> str:
    blob = json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:32]


def get(request: dict):
    p = cache_dir() / f"{key(request)}.json"
    if p.exists():
        return json.loads(p.read_text())["response"]
    return None


def put(request: dict, response, meta: dict | None = None):
    p = cache_dir() / f"{key(request)}.json"
    p.write_text(json.dumps({"meta": meta or {}, "request_key": key(request),
                             "response": response}, indent=1))
    return p
