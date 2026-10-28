"""Spike 1: Meta API smoke test. Hard cap $1 (budget guard). Expected cost ~ $0.01.

Makes at most 3 paid calls: 1 SAM 3.1 image, 1 Spark vision, 1 Spark tool call.
Saves raw responses in the cache so parsers can be written without new calls.

Run (WSL):  MODEL_API_KEY=... python -m spikes.api_smoke <image.jpg> "<sam prompt>"
"""
import json
import os
import sys

from PIL import Image

from backends.meta import budget, client


def main(img_path, prompt):
    print(f"budget ${budget.budget():.2f}, spent so far ${budget.spent():.4f}")
    im = Image.open(img_path)
    im.thumbnail((int(os.environ.get("SMOKE_MAX", 518)),) * 2)
    uri = client.image_data_uri(im, quality=85)

    # 1. SAM 3.1 on one small image.
    s = client.sam31_image(prompt, uri)
    print("\n[sam-3.1] status", s["status"], "raw bytes", len(s["raw"]))
    print(s["raw"][:1500])

    # 2. Spark vision: describe the object.
    v = client.spark_chat([{"role": "user", "content": [
        {"type": "text", "text": f"Look at the {prompt}. Reply with JSON only: "
                                 '{"material": str, "hollow": bool, "mass_kg": float, "friction": float}'},
        {"type": "image_url", "image_url": {"url": uri}}]}], max_tokens=int(os.environ.get("SMOKE_TOK", 120)))
    print("\n[spark vision]", json.dumps(v, indent=1)[:1500])

    # 3. Spark tool call: text only.
    tools = [{"type": "function", "function": {
        "name": "push", "description": "Push an object on the table.",
        "parameters": {"type": "object", "properties": {
            "obj": {"type": "string"}, "dir": {"type": "string", "enum": ["left", "right", "forward", "back"]},
            "force": {"type": "number", "description": "newtons"}}, "required": ["obj", "dir", "force"]}}}]
    t = client.spark_chat([{"role": "user", "content": "Give the mug a gentle push to the left."}],
                          tools=tools, max_tokens=int(os.environ.get("SMOKE_TOK", 120)))
    print("\n[spark tools]", json.dumps(t, indent=1)[:1500])
    print(f"\nspent total ${budget.spent():.4f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "object")
