"""Print cached API responses. For SAM 3.1, prints the stream events that carry output.

Run (WSL): python -m spikes.dump_cache [model] [chars]
"""
import json
import sys

from backends.cache import cache_dir


def main(model="sam-3.1", n=1500):
    for p in sorted(cache_dir().glob("*.json")):
        d = json.loads(p.read_text())
        if d.get("meta", {}).get("model") != model:
            continue
        print(f"== {p.name} meta={d['meta']}")
        r = d["response"]
        if isinstance(r, dict) and "raw" in r:
            for line in r["raw"].splitlines():
                if line.startswith("data:") and any(k in line for k in ('"delta"', "completed", "error")):
                    print(line[:n])
        else:
            print(json.dumps(r)[:n])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "sam-3.1", int(sys.argv[2]) if len(sys.argv) > 2 else 1500)
