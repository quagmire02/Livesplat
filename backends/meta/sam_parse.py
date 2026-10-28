"""Parse a cached SAM 3.1 SSE stream into full-frame boolean masks, using meta-sam-parser.

parse_image(raw_sse_text, (H, W)) -> list of {"object_id", "box" (x1,y1,x2,y2), "conf", "mask" (H,W) bool}
"""
import asyncio
import json

import cv2
import numpy as np
from meta_sam_parser import (SegmentationBoxRecord, SegmentationMaskRecord, decode_mask_to_raster,
                             image_segmentation_format, parse_responses_stream)


def _events(raw: str):
    out = []
    for line in raw.splitlines():
        if line.startswith("data:"):
            try:
                out.append(json.loads(line[5:].strip()))
            except ValueError:
                pass
    return out


async def _aiter(items):
    for it in items:
        yield it


async def _parse(raw):
    stream = parse_responses_stream(_aiter(_events(raw)), image_segmentation_format())
    return await stream.final_result()


def parse_image(raw: str, hw):
    res = asyncio.run(_parse(raw))
    H, W = hw
    boxes = {r.object_id: r for r in res.records if isinstance(r, SegmentationBoxRecord)}
    out = []
    for r in res.records:
        if not isinstance(r, SegmentationMaskRecord):
            continue
        m = np.frombuffer(decode_mask_to_raster(r.mask), np.uint8).reshape(r.mask.height, r.mask.width)
        b = r.bounds
        x1, y1, x2, y2 = (int(round(v)) for v in (b.left, b.top, b.right, b.bottom))
        full = np.zeros((H, W), bool)
        if x2 > x1 and y2 > y1:
            full[y1:y2, x1:x2] = cv2.resize(m, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST) > 0
        bx = boxes.get(r.object_id)
        out.append({"object_id": r.object_id,
                    "box": (bx.left, bx.top, bx.right, bx.bottom) if bx else (x1, y1, x2, y2),
                    "conf": r.confidence, "mask": full})
    return out
