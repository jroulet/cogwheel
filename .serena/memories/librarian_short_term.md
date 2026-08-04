# Librarian Short-Term Observations

## Run: 2026-08-04 — Schwinger quad-double extension (mpmath path for w > 60)

**Scope:** Post-commit sync for commit `2e387c9` (Schwinger QD extension — mpmath path).

**Changed files of doc relevance:**
- `cogwheel/lensing/chang_refsdal/_schwinger.py` — new `_f_schwinger_mpmath()`, `W_CEILING_SCHWINGER_QD = 150`
- `cogwheel/lensing/chang_refsdal/operator.py` — routes 60 < w <= 150 via mpmath
- `cogwheel/lensing/surrogate_training.py` — `_SADDLE_W_CEILING` raised 58 → 148
- `pyproject.toml` — new `training` extra with `mpmath` dependency

**Stale surface found and fixed:**
- `SPEC.md` line 54 (microlensing engine row): Schwinger evaluator refusal ceiling was "w > 60"
  — updated to describe two-tier path (dd up to 60, mpmath 60–150, refuse above 150).
  F019 note extended from "TWO DIFFERENT 60s" to "THREE DISTINCT CEILINGS".

**TODO closed:** `todo.d/schwinger_qd-extension.md` (tagged `[→ spec]`).

**Fragments created:**
- `spec_changelog.d/2026-08-04_schwinger-qd.md` (bump: patch)
- `completed.d/2026-08-04_schwinger-qd-extension.md`

**Surfaces confirmed NOT stale:**
- `docs/source/overview.rst` — no Schwinger/mpmath mentions, no change needed
- `docs/source/installation.rst` — doesn't document pip extras at all; `training` extra with mpmath
  is developer/surrogate-training-only, not worth adding to the sparse install page
- `docs/source/api.rst` — no new modules/subpackages
- `DATA_CONTRACTS.yaml` — no new disk artifacts
- `SPEC.md` `_SADDLE_W_CEILING` reference — only cites the constant NAME, not its value; still accurate

**Stale SPEC pattern this commit reveals:**
- Implementation-level ceiling constants cited by VALUE in SPEC ("above which it refuses by name")
  become stale whenever the ceiling moves. This is the second occurrence (first: min_gamma_band).
  Watch: any SPEC sentence containing "above which it refuses" or a specific numeric threshold.

**Fragile cross-references to watch:**
- `W_CEILING_SCHWINGER_QD = 150` is now cited in SPEC.md; if the constant is renamed or the
  ceiling raised again, the SPEC text + F019 note both need updating.
- `_SADDLE_W_CEILING = 148` (= QD ceiling minus 2) — if the QD ceiling changes, this must track.
- `pyproject.toml` `training` extra — if mpmath becomes a core dependency (not optional),
  installation.rst would need updating.

**Side effects:**
- `sync_derived_docs.py` "auto-fixed" with no real diff — confirmed no-op (tidy_advisory.json
  was the only stray; reverted before commit).
- Consumer graph warnings (test-only callers of `LensAmplificationSurrogate.load`) are
  pre-existing, test-only — correctly left off DATA_CONTRACTS.yaml per convention.
