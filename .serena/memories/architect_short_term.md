# Architect Short-Term Observations

## Build 8b-levers — FINAL plan emitted 2026-07-20

Two Coder WPs (Simplifier: keep split). WP-A geometry Newton caustic
shortcut, WP-B operator contraction fusion. Professor rulings encoded:
(Q1) HEAD_NEAREST_CAUSTIC_PINS bit-exact theta -> <=1e-10 value-
preservation gate (legit re-cert; distance stays assertEqual/places=14);
routed to Test Dev. (Q2) 1-D scalar Newton on g'(theta)=0, analytic
g'/g'', 32-pt coarse seed, 2 best cells, MANDATORY single-cell Brent
fallback, g''>0 guard, wedge-clamp per lobe/branch, seed-per-lobe
take-min. (Q3) 9 saddle branch/lobe configs at 1e-10 both parities.
(Q4/Q5) fusion = dispatch-only njit merge preserving accumulation order,
NO reassociation, byte-exact re-cert, half_sum stays arg +
_SERIES_TOLERANCE module-global for F010. has_domain_changes=true,
has_spec_update=true.

(empty — last consolidated by Dreamer on 2026-07-20)

## Build 8c-cont triage — 2026-07-20
INS-1-001 (missing census test suite, confirmed absent via find_file/grep) ->
coder_fix, routed to Test Developer (never Coder) sharded ~9 specs per
Inspector's routing, reconciled VERBATIM vs build8c_plan_approved.json.
INS-3-001 (dropped_gamma_slivers not threaded into saved-artifact
provenance, confirmed via code read: _build_provenance omits it, census's
own _dropped_slivers_from docstring already flags the discrepancy) ->
coder_fix, BUT corrected the Inspector's proposed shape after Simplifier
review: must be a FLAT list of [lo,hi] pairs (matching
_normalize_slivers/dropped_slivers_from_training_report's existing flat
contract), NOT a dict keyed by parity-label as literally suggested — a
dict shape breaks/crashes the existing `for lo, hi in dropped_slivers`
consumer in surrogate_census.py. Lesson: always feed Inspector-authored
fix snippets through Simplifier before endorsing verbatim — the shape
mismatch would have shipped a fix that passes TEST 12 in isolation while
breaking the default provenance-read path it exists to fix.
INS-1-003 (SPEC/doc staleness) -> override, doc-sync is post-gate driver
responsibility per project policy regardless of what any single build's
plan says.
