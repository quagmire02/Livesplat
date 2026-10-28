# agent/ — tools, planner, controller
Files:
- schema.py — 16 tools (OpenAI function format, also sent to Spark). validate() checks names, ranges, enums.
- controller.py — Runs tools. Finds objects by name (names.json, then automatic names). Directions are relative to the camera. reset is FULL: poses, colours, sizes, removed objects, properties, gravity.
- parser.py — Local rules (free). normalize() changes synonyms and typos. "what if/would/will" gives simulate. "it" = last object or the selected object.
- planner.py — handle(): local = parser; meta = Spark tool loop (max 3 rounds). The multi-round path is not tested with paid calls.
- properties.py — Spark vision, one crop per object. Writes names.json and props.json. Paid.
Status: 28 tests pass. Spark round 1: names correct, planner 4/5.
