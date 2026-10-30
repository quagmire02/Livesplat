# agent/ — tools, planner, controller
Files:
- schema.py — 19 tools (incl. slide; plan_goal and explain are handed to the viewer: director / viewer.explain) (OpenAI function format, also sent to Spark). validate() checks structure, types, finite ranges, enums and bounded simulation actions.
- controller.py — Runs tools. Finds objects by name (names.json, then automatic names). Directions are relative to the camera. reset is FULL: poses, colours, sizes, removed objects, properties, gravity.
- parser.py — Local rules (free). normalize() changes synonyms and typos. "what if/would/will" gives simulate. "it" = last object or the selected object.
- director.py — Director mode: Muse proposes plans for a goal, MuJoCo simulates each on a copy of the state, Muse scores measured outcomes and revises; best plan plays live. Logs to runs/<scene>/director/.
- planner.py — handle(): local = parser; cached = strictly cached Muse responses; meta = approved Spark tool loop (max 3 rounds). Selection grounds "it". Nullable tool messages and read-tool/action follow-ups have mocked tests; no new paid validation.
- properties.py — Spark vision per object: names.json, props.json (material, mass, friction, bounce, sam_prompt, real_size_cm, size_confidence) and scale.json (metres per unit = median real/measured height). Paid.
- tag.py — `python -m agent.tag <run_dir>` = properties (Spark) + segment.sam_vote (SAM 3.1). recon.build runs it only with --tag. Cache misses need an explicitly enabled, approved API round.
Controller predictions use a separate MuJoCo data state, restored even on errors; yaw is not counted as tilt. Measurement reports scene units until calibrated. Object IDs are excluded from quantity parsing.
Status: 28 tests pass. Spark round 1: names correct, planner 4/5.

- [CHANGE_NOTES.md](CHANGE_NOTES.md) — Short implementation summary and remaining gaps (Oct 29).
