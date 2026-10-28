# agent/ — tools, planner, controller
Files:
- schema.py — TOOLS (OpenAI function format, sent as-is to Spark), COLORS, validate(call) with numeric limits; simulate may only contain push/drop/throw/stack.
- controller.py — Controller(scene, world, run_dir, on_appearance, camera): resolves names (names.json first, then auto: colour, "object k", "tallest"), camera-relative directions, runs tools t_*. simulate saves/restores physics state.
- parser.py — local rule planner: normalize() (synonyms, typo fix via difflib, "everything"), clause rules; splits on , ; then also, and+verb. "what if/would/will" -> simulate. "it" -> last object or GUI selection. Unknown clauses get a hint.
- planner.py — handle(text, ctrl, backend, default): local -> parser; meta -> Spark tool calling (needs LIVESPLAT_ALLOW_META=1 in the viewer + user approval).
Tools: list_objects, measure, push, drop, throw, stack, set_mass, set_friction, set_bounce, set_gravity, set_color, scale, remove, restore, reset (FULL: poses, colours, sizes, removed, properties, gravity), simulate.
Names: <run>/names.json {id: [names]} — fixture for table1 in tests/fixtures; Spark vision will write it later.
Status: 15 tests pass on table1. Unknown requests are reported, never guessed.
