"""Baseten test: 1 text answer, 1 tool call, 1 vision call (object crop). Run with LIVESPLAT_ALLOW_BASETEN=1."""
import json
import sys

from PIL import Image

from backends import baseten
from backends.meta import client


def show(tag, r):
    m = (r.get("choices") or [{}])[0].get("message") or {}
    print(f"\n[{tag}] status {r.get('_status')} usage {r.get('usage')}")
    print("  keys:", sorted(m))
    print("  content:", (m.get("content") or "")[:400])
    if m.get("tool_calls"):
        print("  tool_calls:", json.dumps(m["tool_calls"])[:400])
    if r.get("error"):
        print("  error:", str(r["error"])[:400])


def main(crop):
    show("text", baseten.chat([{"role": "user", "content": "In one sentence: what is a Gaussian splat?"}], max_tokens=800))
    tools = [{"type": "function", "function": {"name": "push", "description": "Push an object on the table.",
              "parameters": {"type": "object", "properties": {"obj": {"type": "string"},
                             "direction": {"type": "string", "enum": ["left", "right", "away", "toward"]}},
                             "required": ["obj", "direction"]}}}]
    show("tools", baseten.chat([{"role": "user", "content": "give the cetaphil a little shove to my left"}],
                               tools=tools, max_tokens=1500))
    im = Image.open(crop)
    im.thumbnail((512, 512))
    show("vision", baseten.chat([{"role": "user", "content": [
        {"type": "text", "text": 'Reply JSON only: {"name": str, "visible_text": str, "material": str}'},
        {"type": "image_url", "image_url": {"url": client.image_data_uri(im)}}]}], max_tokens=1500))


if __name__ == "__main__":
    main(sys.argv[1])
