# Foreman Short-Term Observations

## 2026-08-19 (tiling_plan_refresh_order16)
- TILING-PLAN DEMAND GATE IS BINARY, SO CENSUS SHRINK ≠ PLAN SHRINK: the
  `tiling_plan` demand gate admits a (region, gamma-band) band whenever the
  census cell's `engine_residual` count > 0 (`_residual_by_region_band`), and
  sizes it from the measured w-range of those residual draws. After the
  order-16 diffractive certificate fix the 10k census `engine_residual` share
  fell 42.06% -> 33.15% (routes: diffractive_analytic +812, born_analytic
  +79, engine_residual -891), but the plan came out BYTE-IDENTICAL
  (11252 nodes / 90016 calls): the exterior:+1 cell still has residual
  demand in the same measured w-range (0.0248-4.97), so its 19-tile / 6080-
  node plan persists and exterior:+1 is still 54.0% of planned nodes ->
  escalation does NOT clear. Lesson: an escalation cleared by reducing
  residual COUNT must be re-checked against the demand GATE, not assumed.
- Long census runs (order-16, ~1.1 draws/s, 10k = ~2.4h): `serve_route_census.
  run` prints nothing, and Serena's `execute_shell_command` + the opencode
  Bash tool both hang on the background process (tool waits on the child's
  stdout). Working recipe: `setsid nohup python script > log 2>&1 < /dev/null
  &` then poll the log with short foreground Bash calls — the setsid detach
  is what lets the Bash tool return. To get per-250-draw progress, monkeypatch
  the module-global `classify_draw` in a driver BEFORE calling the shipped
  `tiling_plan.run` (wrap the shipped predicate, don't re-type it) — that
  keeps mirror-fidelity while emitting draw-level progress.
- NUMBA warm cache: `/tmp/numba_census_shared` is the shared cache the brief
  prescribes (`NUMBA_CACHE_DIR=...`); first census in a fresh shell still pays
  ~10-15s warmup but steady-state was ~1.1-1.2 draws/s at order 16 / n_freq
  128.
- Parallel-session contamination on a long run: the tree was clean at launch
  (14:53) but had 6+ source files modified by a PARALLEL build session by
  17:11 (timestamps inside my run window). My census ran against HEAD-as-loaded
  at process start, so results are valid; verify with mtime ordering before
  attributing blame.
- 10k order-16 refresh vs the brief's 3k measurement: engine_residual 3315/
  10000 = 33.15% vs 3280/10000 scaled from 3k (32.80%) — +0.35 pct-pts, within
  the 1-2 pct-pt allowance; all six major routes within ~1 pct-pt. Acceptance
  bars (within 1-2 of 32.80%, clearly below 42.06%) PASS.
- `render_fragments.py` emits "Repoint to the completed.d record" WARNINGS for
  dangling wiki-links in unrelated fragments — pre-existing, exit 0, TODO.md
  still regenerates. Not caused by my edit.