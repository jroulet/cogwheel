# Librarian Short-Term Observations

## Run: 2026-08-13 — post-commit sync, primary c0da6cb (tier-1 saddle analytic rung)

**Scope**: `.claude/sync_issues.json` listed ~20 pending commits back through
`5fbd024`. Triaged: all but three touch only `.claude/spec/`, `.claude/handoff/`,
`.claude/agent_state/`, `.serena/memories/` — each already self-rendered
(TODO.md/COMPLETED.md changed in the SAME commit) — genuine no-ops for doc
surfaces, confirmed by pattern (agent-authored spec fragments are the
authoritative record, not something Librarian propagates further).

Code-touching commits:
- `3f4106e` (`scripts/census_dry_run.py`, new file) — a GEOMETRY-ONLY
  structural dry-run script with its own duplicated `_classify_saddle`
  (does not reuse `surrogate_census.characterize_sample`). No disk-persisted
  artifact, console output only -> SCRIPTS/ REWRITE NO-OP RULE applies
  (long-term memory). Skipped.
- `4e724097` (`cogwheel/lensing/surrogate_training.py`, degenerate exterior
  band recording) — training-report diagnostic addition (`chart_<label>_
  exterior_band_degenerate` record), fully self-documented in its own
  completed.d fragment, same family as the already-SPEC'd `beyond_w_cap`
  loud-recording convention. Judged too deep/narrow for a SPEC sentence
  (fragment itself says "NOT settled here, left open deliberately" —
  design still in flux) and not the driver's flagged primary. Skipped as
  proportionate; flag for a future pass if this recording pattern becomes
  load-bearing.
- `c0da6cb` (PRIMARY) — new tier-1 far-from-caustic macro-saddle analytic
  serve rung. Verified independently against code before editing:
  - Read `LensedRelativeBinningLikelihood._amplification_coefficients`
    (likelihood.py ~2129-2271) in full: confirmed dispatch order is
    surrogate intercept -> ppGO above-ceiling (`w_max > W_CEILING_SCHWINGER_QD`)
    -> tier-1 saddle analytic (`gamma > 1`, via `_saddle_farfield_analytic`)
    -> exact seed engine. Matches driver's context exactly.
  - Confirmed `_gauge.py`'s new `_saddle_switch_delay`/`_saddle_phase_delay`
    have NO production callers (grepped, only referenced in their own
    docstrings and the tier-1 function's docstring disclaiming use of them).
  - Confirmed `surrogate_census.py` imports `_saddle_farfield_analytic_serves`
    and `characterize_sample` gained a `'saddle-farfield-analytic'` served
    category — but per the build's own todo fragment
    (`lensing_saddle_tier1_cannot_reach_the_census_gap.md`), this category
    is currently UNREACHABLE in production census runs (the `rho > 1 -> born`
    routing upstream means saddle draws never carry `rho >= 2`). Chose NOT
    to add this category to SPEC's CENSUS 7-way-breakdown sentence (unlike
    `ppgo_fold`, which fires and is genuinely observed) — a category that
    provably never populates would misrepresent the breakdown as richer
    than it behaves; the disjointness is already the single most important
    fact of this build and I said it in the new dispatch-order sentence
    instead. Revisit if a later build makes the routing order-independent.
  - No new disk artifact confirmed (checked `_saddle_farfield_analytic` body
    — reconstructs via `switched_analytic_channels` with zero envelope,
    nothing serialized) -> DATA_CONTRACTS.yaml needs NO new entry; existing
    `surrogate_census.py` consumer row (function: `run`) stays accurate,
    the new import is a code dependency not a data-contract concern.
  - FINDINGS.md F066-F068: read in full, sequential after F065, no dangling
    cross-refs, self-contained process/methodology findings (mutation-probe
    binding trap, escalation turn-budget, DRY-vs-leaf-isolation tension) —
    no action needed.
  - `docs/source/` has zero mentions of any saddle/gauge symbol from this
    build — confirms (again) the lensing surrogate/engine internals are
    correctly absent from the Sphinx narrative.

**Fixed**: SPEC.md, one paragraph extended (row 55, "Microlensed waveform &
likelihood"): (a) reworded "opens with a ppGO rung" -> "dispatches, after the
surrogate intercept, a ppGO rung" (was misleadingly implying ppGO is first);
(b) appended a new TIER-1 FAR-FROM-CAUSTIC MACRO-SADDLE ANALYTIC INTERCEPT
sentence: full dispatch order, two-term gate, measured accuracy, and an
EXPLICIT non-improvement statement ("does not move structural coverage") per
the driver's fact 5 — deliberately did NOT quote the literal 87.61% figure,
since SPEC.md has never quoted a coverage percentage anywhere (grepped, zero
hits) and introducing the first-ever number there is a bigger narrative
commitment than this sync warranted; the qualitative disjointness statement
carries the same "do not read this as an improvement" force without minting
a number SPEC would then need to keep in sync with every future tier's
measurement. `spec_changelog.d/2026-08-13_saddle_tier1_analytic_rung.md`
(bump: patch, precedent: other single-new-rung additions are patch in this
repo). Rendered via `scripts/render_fragments.py` (0.37.16).

**Skipped / no-op, with reason** (see above): `3f4106e`, `4e724097`, all
spec/handoff-only commits, DATA_CONTRACTS.yaml (no new artifact), docs/source
(zero references), `_gauge.py` tier-2 helpers (correctly left undocumented —
build's own docstrings already say "no production caller yet", nothing to
sync).

**Process note**: `scripts/render_fragments.py` again left a stray diff in
`.claude/tidy_advisory.json` (matches the long-term-memory rule) — reverted
with `git checkout --`. Confirmed the Serena-heredoc-silent-noop trap once
more: a `python3 - <<'EOF' ... EOF` command via `execute_shell_command`
printed nothing (stdout empty, rc 0) even for a two-line `print()` script —
switched to writing the script to a file (Write tool) and running
`python3 <path>`, which worked immediately. Do this from the start next time
instead of re-discovering it.
