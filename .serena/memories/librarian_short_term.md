# Librarian Short-Term Observations

## Run: 2026-08-04 — train_surrogate_production.py addition (commit c5b8a4a)

**Scope:** Post-commit sync for commit `c5b8a4a` (scripts: production training
launcher — DD band, w <= 60).

**Changed file:** `scripts/train_surrogate_production.py` — new 94-line launcher.

**Triage result:** SCRIPTS/ REWRITE NO-OP RULE applies. Full justification:
- Script stays entirely in `scripts/`, never touches `cogwheel/`.
- Output path `OUTDIR = "/tmp/surrogate_production_dd"` is transient; not a
  committed artifact. `train()` from `cogwheel.lensing.surrogate_training`
  does the actual serialization (tracked separately if at all in DATA_CONTRACTS).
- No `cogwheel/` public API changes.
- No dependency/install changes.
- Not a notebook or test change.

**Surfaces confirmed NOT stale:**
- SPEC.md — no reference to training launcher; launcher is config-only.
- DATA_CONTRACTS.yaml — no new committed disk artifacts.
- docs/source/ — scripts/ is not a cogwheel module; no API page needed.
- lensing_coverage_map.md — Section D already records "Production training can
  proceed on the current architecture" (written 2026-08-04 after census run).
  The launcher's existence doesn't require a new note in the coverage map.
- TODO fragments — no fragment references a "production launcher needs to be
  written" as an action item. Script is a standalone artifact.
- FINDINGS.md — no new finding, no old finding invalidated.

**Action taken:** No doc changes. Deleted `.claude/sync_issues.json` (gitignored).

**Stale pattern this commit reveals:**
- Pure-launcher scripts in `scripts/` (even with `/tmp/` output paths) are
  routinely no-ops for doc sync as long as they don't commit new artifacts.
  The key distinction: does the script CREATE a committed artifact or call an
  existing function that already owns the artifact contract?
