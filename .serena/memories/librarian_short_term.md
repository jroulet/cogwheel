Post-commit sync 2026-08-14, --post-commit (backlog: 55373b2..caf6026, 13
commits, ending at the c3-certificate saddle-admission build
`symmetry_tie_c3_admission` / 1c90b3a + its TODO-closure churn). Serena MCP
was UP and worked normally throughout (no fallback needed this run).

Step 0: `sync_derived_docs.py` -> "5 checks run, all OK", zero diff.
`render_fragments.py` first pass -> "All surfaces up to date" (only the
5 pre-existing FINDINGS-Fxxx false-dangle wiki-links, already tracked, not
re-noted). Reverted a `.claude/tidy_advisory.json` side-effect diff from
running render_fragments.py (known no-op artifact, per prior memory).

MAIN FINDING — driver had ALREADY correctly handled the doc-facing side of
the c3-certificate build before I started: SPEC.md's tier-1 saddle
paragraph (constants `_SADDLE_FARFIELD_SAFETY=20`,
`_SADDLE_FARFIELD_CERT_BAR=1e-3`, `_SADDLE_FARFIELD_MIN_IMAGE_SEP=0.05`,
function `_saddle_farfield_analytic_serves(real_images, source, matrix,
w_lo)`) verified byte-accurate against `cogwheel/lensing/likelihood.py`
(lines ~201-223, ~555-621, ~2105-2114). `surrogate_census.py`'s census
mirror verified to import and call the SAME
`_saddle_farfield_analytic_serves` (not a re-derived rho gate) — matches
SPEC's "census-mirror equality" claim. `spec_changelog.d/
2026-08-14_saddle_certificate_gate.md` has no stub marker, sane minor bump.
docs/source/*.rst has ZERO hits for "saddle" or even "lensing" anywhere —
this Sphinx site does not document the lensing subpackage at all (confirmed
`api.rst` uses bare `:recursive:` autosummary over `cogwheel`, so no manual
per-module entry is ever needed there) — reconfirms the standing pattern,
nothing to touch downstream for this build.

Verified the repeated "close TODO 2/7 saddle_admission_c3" commit chain
(fc99aee -> 1b79cbc -> 40bf814 -> fb93710 -> caf6026, all same timestamp
batch) is NOT duplicate/error churn: fc99aee wrote
`completed.d/2026-08-14_saddle_admission_c3.md`, 1b79cbc renamed it (0
content diff) to `2026-08-14_lensing_saddle_admission_c3.md` to match the
todo.d stem convention, and the remaining three commits are sequential
`depends_on:` dangling-repoint fixes on `lensing_wire_serving_artifacts.md`
/ `lensing_certified_map_guard_relaxation.md` / `lensing_training_campaign.md`
as the c3 todo item retired out from under them — legitimate, no
double-fix. `git diff` between each consecutive pair confirmed one small
real change per commit, never a no-op re-run.

F079 (new finding, `_find_cusps` wrap-arithmetic half-ring bug) cross-checked:
both fragments it names resolve
(`todo.d/lensing_find_cusps_wrap_bug.md`, `todo.d/
lensing_cusp_arm_coverage_constant_stale.md`), both still correctly describe
the retirement of `_CUSP_ARM_COVERAGE`/measure_cusp_arm_*.py scripts as FUTURE
work (matches the brief's note that those scripts still exist today — do not
flag as stale, they are pre-fix).

NEW REAL FINDING, filed as `todo.d/
lensing_calibrate_saddle_exterior_certificate_broken_import.md`:
`scripts/calibrate_saddle_exterior_certificate.py` (added by the driver
calibration commit `a4ba536`, immediately BEFORE `1c90b3a` deleted its
import target) does `from cogwheel.tests.test_lensing_saddle_tier1_accuracy
import (W_FLOOR, _exact_total_w, _min_delta_tau, _polar_source,
_tier1_serve)` — that test module no longer exists (retired by 1c90b3a,
successor is `test_lensing_saddle_serve_gate.py` under new helper names).
The script's own docstring still claims "at HEAD" it's the live-importable
one and that `measure_saddle_eta_floor.py` (also since deleted) is the
stale one — both claims are now backwards. NOT edited (scripts/*.py is
code, out of Librarian scope — "no code edits" hard rule applies beyond
just cogwheel/). Not currently blocking anything: the script already
produced its output (`scripts/calibration_pilot_followup.json`, correctly
cited in SPEC.md as calibration provenance, not a served DATA_CONTRACTS
artifact per the brief's explicit judgment call) before going stale, so no
one is silently trusting a broken script. NEW CROSS-REFERENCE PATTERN TO
WATCH: a driver-authored one-shot calibration script that imports test
helpers directly is fragile against the VERY NEXT commit deleting those
helpers if the calibration and the build that retires the old gate land
back-to-back — check for this when a `calibrate_*`/`measure_*` script's
import block names a test module and the adjacent build touches that same
test module.

Committed as `docs: post-commit sync (...)`, staged ONLY
`.claude/spec/TODO.md` + the new `todo.d/` fragment (my memory writes too).
Deleted `.claude/sync_issues.json` per protocol.
