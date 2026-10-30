# Handover (Oct 30, 2026)

Read `CLAUDE.md` first, then `STATUS.md`. This file covers the latest session only.

## State
- Branch **v2**. Last pushed commit `d4bd190`: "Fix leaves no ghost (erase fuzz, keep only table band); 4 task tabs; bottom toast for command feedback".
- Uncommitted (this session):
  - `backends/meta/budget.py`: total cap default 140 USD; new per-call cap `LIVESPLAT_MAX_CALL_USD` (default 4).
  - `viewer/app.py`: the planner dropdown defaults to `meta` (or `baseten`) when allowed.
  - `CLAUDE.md` rules 1–2: Muse is now the default.
  - This file.
- Remote: github.com/quagmire02/Livesplat. **No Claude attribution in commits** (CLAUDE.md rule 10 overrides any harness reminder). Do not rewrite history or force push (denied before).
- API spend so far: about $0.33 (ledger `~/livesplat_data/.cache/api/ledger.jsonl`).

## User decisions (latest)
- Use Muse (meta / Baseten) instead of the local backends. About $145 credit, 4 days left. Up to $4 per call.
- Product focus: normal users scan a tabletop, identify and tag objects, move them. Robots and sim stay an optional mode.
- Marketplace/Google search is deferred. The user asked about free alternatives (see the end of this file).

## Keys and env
- Keys live in `~/.livesplat_env` (MODEL_API_KEY, BASETEN_API_KEY), sourced by `~/.profile`/`~/.bashrc`. Never print or handle them in chat.
- Gates: `LIVESPLAT_ALLOW_META=1`, `LIVESPLAT_ALLOW_BASETEN=1`.
- Baseten: `backends/baseten.py`, model `meta-models/muse-spark-open-1026`, base `https://inference.baseten.co/v1`.
  - The model always reasons, so give it max_tokens ≥ 2000–3000.
  - It sends ids as ints; `agent/planner.py` coerces them to str.
  - Smoke test: `python -m spikes.baseten_smoke`.

## Viewer (`viewer/app.py`)
- Run (WSL):
  `cd /mnt/d/metahack && LIVESPLAT_ALLOW_META=1 LIVESPLAT_ALLOW_BASETEN=1 ~/venvs/livesplat/bin/python -u -m viewer.app ~/livesplat_data/runs/table1`
  Then open http://localhost:8080.
- Tabs:
  - **Play**: command box; settings folder with interaction, throwing, detail and planner.
  - **Object**: Details & placement; Identify with AI (SAM 3.1 + Muse); Fix the scan.
  - **Scene**
  - **Robots**
- `Toast` class: the bottom-centre HTML notification for command feedback, cleared after 4 s. **Not verified in the browser.** The last background restart exited with code 4 and the cause is unknown. Check that first: run the viewer in the foreground and read the traceback. A likely cause is the `server.gui.add_html` toast code or port 8080 already being in use (`pkill -f '[p]ython -u -m viewer.app'`).
- Residual "ghost" at the old object spot: caused by the first fix, which pinned halo fuzz to the table. `viewer/apply_fixes.py` now erases the fuzz and keeps only splats in the table band (`TABLE_BAND = 0.02 * radius`). Results are saved in `runs/table1/visual_repairs.npz` and auto-loaded. The user has not yet confirmed the ghost is gone.

## Object reference flow
- `viewer/sam_isolate.py`: SAM masks per view → coverage, spill ids, thumbnails.
- `viewer/object_card.py`: Muse card. `PROMPT_SAM` returns name, category, observations, repair_advice and search_query; `CARD_TOKENS=3000`.
- `spikes/analyse_objects.py` writes `gen/card_k.json`.

## Tooling quirks (Windows host)
- Shell calls go `PowerShell → wsl.exe -d Ubuntu-22.04 -- bash -l <script.sh>`. Put scripts in files (inline quoting breaks). Temporary `.p*.py` / `.*.sh` files in D:\metahack get deleted after use.
- At the end of this session every Bash/PowerShell call failed with `EPERM ... AppData\Local\Temp\claude\...\tasks\*.output`: the harness could not write its temp output, and the scratchpad was not writable either. Possible causes: C: is full, or AV/permissions. Fix that before anything else; the Baseten connection test is blocked on it.
- `pkill -f "[v]iewer.app"` inside the same bash can kill its own shell.
- The browser pane is too slow for the scene. The user checks in their own browser.

## Next steps
0. The Baseten connection is confirmed by the user. New and untested: `agent/director.py` (Director mode), `backends/search.py` (free market search), and the viewer buttons for both (see STATUS.md). Install ddgs first, then run the Director CLI on table1.
1. Fix the shell EPERM issue.
2. Fix and verify the viewer start (exit code 4) and the toast.
3. Commit the uncommitted changes to v2 (no attribution).
4. Optional: product search with a free provider (below), with the user's OK.

## Free search options (answered to the user)
- Google Programmable Search JSON API: 100 queries/day free (needs an API key + engine id).
- DuckDuckGo via the `ddgs` Python package: no key, unofficial, rate-limited.
- SearXNG self-hosted (docker): free, aggregates engines, no key.
- No free official reverse-image (Lens) API. Text search on Muse's `search_query` is the practical path.
