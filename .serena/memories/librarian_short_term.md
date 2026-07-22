# Librarian Short-Term Observations

## 2026-07-22 — Build 8g far-field tiling + eps gate (SPEC sync)

Scope: sync SPEC.md's surrogate-training narrative after commits 9d3bf90
(Build 8g: far-field tiling, eps registration gate, saddle-tube curvature
guard) and e91550e (max_farfield_regions -> None default).

What went stale: the GLOBAL MULTI-CHART ARTIFACT sentence said "raw-coordinate
FAR-FIELD charts per image-count region" (the legacy single hard-coded box
per parity, `box_center = (caustic_reach + eta_max + 0.2, 0.0)`) and the
TRAINING sentence had zero mention of far-field placement, the eps gate, or
the foot-of-normal tube skip. Fixed both in place (present tense); TRAINING
paragraph now has three sub-sections: the pre-existing FOOT-OF-NORMAL guard
sentence (folded in, not new to 8g historically as a mechanism but its
report-recording behavior IS 8g), FAR-FIELD TILING (Build 8g), REGISTRATION
GATE (Build 8g).

Ground-truthed every claim against actual code (get_symbols_overview then
find_symbol include_body=True on `_mass_strata`, `_stratum_w_range`,
`_farfield_tiles`, `TrainingConfig`, `_min_curvature_radius`, `_chart_gated`,
`_gate_chart`, `_load_or_build`, `_train_band_charts`) rather than trusting
the caller's or the todo.d fragment's framing. Confirmed via
`git log --oneline -- cogwheel/lensing/surrogate_training.py` that all three
(tiling + eps gate + saddle-tube curvature guard) landed together in 9d3bf90
("Build 8g — far-field tiling, eps registration gate, saddle-tube guards"),
and the max_farfield_regions default flip is the separate followup commit
e91550e — so the todo.d fragment's item (3) "diagnose and fix the saddle
tube tail" resolved as: root cause = curvature-radius foot-of-normal skip
(training-time) PLUS the eps gate (serving-time exclusion of the 0.43/1.15/
2.15 bad charts) — not a distinct third mechanism. Don't be fooled by a
todo fragment's 3-item list into hunting for a 3rd separate code path when
2 mechanisms cover it.

docs/source/overview.rst: grepped for far-field/farfield/box_center/
surrogate/TRAINING/training — zero hits. overview.rst never described
surrogate training placement at all, so nothing to fix there; confirmed
by search, not skipped on assumption.

Fragile cross-reference: TrainingConfig's own docstring (lines ~226-270)
already carries thorough prose for max_farfield_regions/n_farfield_tiles_per_side/
tube_eps_max/farfield_eps_max provenance (including the WHY behind 5e-2 and
3e-3) — SPEC.md's TRAINING paragraph is a compressed echo of that docstring;
if TrainingConfig's defaults or its docstring rationale change again, check
here first for staleness before re-deriving from the functions.

Surprise: render_fragments.py's SPEC_CHANGELOG.md renumbering on this run
was NOT the previously-flagged out-of-order quirk — inserting the new
2026-07-22 fragment cleanly cascaded +0.01 through all entries above the
correct chronological insertion point (Build 8g slotted between 8f=0.7.0
and Build 8a=0.9.0->0.10.0 chain), landing at 0.8.0. Chronologically
sane this time; the alphabetical-vs-date quirk noted in long-term memory
can still bite when filenames don't sort in date order — this one happened
to.

sync_derived_docs.py flagged 4 pre-existing `lens_amplification_surrogate`
consumer_graph warnings (test files not in DATA_CONTRACTS.yaml) — unrelated
to this task's scope (no code changed), left untouched; flagging here in
case a future Librarian pass is asked to reconcile DATA_CONTRACTS.yaml
consumers.

Serena MCP connection dropped mid-task (all mcp__serena__* calls timed out,
then "Connection closed") and reconnected ~2 tool-call-cycles later on its
own. Fallback attempted: raw `Read`/`Bash find` on .serena/memories files —
blocked by the project's own hook ("USE SERENA for project files" / "USE
SERENA for shell commands"), so there was no usable fallback; the only
option was to retry serena tools until reconnection. Worth remembering:
don't burn time hand-rolling a Bash/Read fallback when the hook blocks it
anyway — just retry serena.
