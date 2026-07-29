## 2026-07-29 (later still) post-commit sync — build 1a analytic caustic derivatives (commit 1a82046)

Scope: 11 pending commits in `.claude/sync_issues.json` (842176e..1a82046).
The first 10 were spec/handoff/memory-only (confirmed by `git show --stat
--name-only <c> | grep cogwheel/.*\.py` returning empty for every one) —
skip-entirely per house triage table, matches driver's framing. Only 1a82046
had a real `cogwheel/**` diff: four new public functions in the EXISTING
`cogwheel/lensing/chang_refsdal/geometry.py` (`caustic_derivatives`,
`caustic_speed`, `caustic_curvature_radius`, `fold_opening_direction`).

FIXED:
1. SPEC.md's giant "Microlensing engine" row (line 53, 0-indexed 52): the
   `geometry.py` parenthetical previously listed only "quartic solver,
   delays, magnifications, stationary-phase kernels, `nearest_caustic_point`"
   — appended the four new names as "analytic closed-form caustic
   derivatives". Note: this same parenthetical never named `r_caustic` either
   — it's a curated highlight list, not exhaustive; matched its existing
   style rather than trying to make it exhaustive.
   `spec_changelog.d/2026-07-29_analytic_caustic_derivatives.md` (bump:
   minor) — rendered to `0.25.0` in SPEC_CHANGELOG.md, NOT the newest-looking
   number, because render_fragments.py bumps by filename ALPHABETICAL order
   within spec_changelog.d/, not content chronology (same quirk as every
   prior session — "analytic_..." sorts before "authoritative_...",
   "fold_arm...", "operator_series...", "spec_rows..." despite being written
   last). Flagged, not fixed, per house convention.
2. `todo.d/lensing_caustic_relative_coordinates.md` step 1's "1a." sub-bullet
   (the cascade spec + its ACCEPTANCE citing F038's OLD 42-case/4.4e-13
   number) rewritten in place to "1a. DONE (2026-07-29, commit 1a82046)"
   with the ACTUAL shipped numbers from the commit message / re-confirmed
   verbatim in FINDINGS.md F038 (4.39e-13 y', 2.56e-14 y'', 110 configs,
   two-stage oracle) — the old acceptance text predated the F038 circularity
   fix (c6c0ec6) and cited stale numbers; correcting these while striking is
   IN SCOPE (not a separate edit) since the whole point of striking is to
   state what's now true. 1b/1c sub-bullets untouched, still pending.
   `lensing_analytic_derivatives.md` needed NO edit — its own "1a" mentions
   ("build 1a exports y'/y''", "1a delivers only the first two orders") were
   already written prospectively in present tense and remain accurate now
   that 1a shipped; nothing there claimed 1a as future/pending in a way that
   needed striking. Don't assume both fragments named in a driver brief need
   symmetric edits — verify each independently.
3. `completed.d/2026-07-29_analytic_caustic_derivatives_1a.md` — new
   completion record, numbers cross-checked against FINDINGS.md F038 (not
   just the commit message) before writing.
4. `changelog.d/2026-07-29_analytic_caustic_derivatives.md` — new public-API
   entry.

VERIFIED, NOT touched:
- `docs/source/api.rst` uses bare `cogwheel` + `:recursive:` autosummary —
  confirmed AGAIN (per standing memory) that new functions in an EXISTING
  module need no manual api.rst entry.
- `docs/source/overview.rst` — zero hits for `geometry.py`/`caustic`/
  `r_caustic`; pitched at architecture level, nothing to propagate (same
  pattern as prior sessions: SPEC gains implementation detail, overview
  doesn't).
- `DATA_CONTRACTS.yaml` — zero hits for any of the four new names; correctly
  so, they're pure in-memory numpy-returning functions, nothing disk-backed.
- FINDINGS.md F038/F039/F040 headings exist and resolve; the file's only
  wiki-link (`[[lensing_caustic_relative_coordinates]]`, inside F038) points
  to a fragment that exists. No Sphinx rebuild needed (no docs/source files
  touched).

Mechanics: `create_text_file`/`replace_content`/`Write`/`Bash cat` are all
HOOK-BLOCKED on this repo for project files — the hook explicitly names the
correct Serena tool per case (even `Write` to a fresh path under
`changelog.d/` got redirected to `mcp__serena__create_text_file`, unlike a
sibling `.claude/spec/` path which the plain `Write` tool silently allowed —
inconsistent enough that the safe move is Serena tools by default for every
project file, not just `.claude/spec/`). `git checkout --` and other read-only
git/gh/conda invocations DO work via Bash directly per the hook's own
exception list. `render_fragments.py` again left a stray
`.claude/tidy_advisory.json` diff (commit-hash/timestamp/touched_files churn)
— reverted via `git checkout --`, not committed, matching every prior
session's note.
