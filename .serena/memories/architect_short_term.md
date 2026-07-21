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

## Build 8e — cusp fast-serving planning — 2026-07-21
Two serving holes to close for "ms everywhere": (1) 8c cusp exclusion
windows (sqrt(eta) fold model invalid, 2/3 Pearcey scaling); (2)
unresolved-high-w corner w>60 non-geometric SchwingerCertificationError
(F019, ~25% prior draws). Owner direction: UNIFORM ASYMPTOTICS — Airy
(fold) + Pearcey (cusp) arms, refusal-conservative fall-through, NO new
exception classes, engine internals untouched, arms are NEW modules +
dispatch-ladder edits. Serving ladder: surrogate->geometric->uniform
(certified)->Schwinger exact(w<=60)->named refusal (measured-hard core).
Corner-scoping census FIRST (fractions a-d Wilson). census script =
scripts/census_homogenization_corners.py (extend). Charts: surrogate.py
TubeChart.cusp_windows, select_chart/_tube_serves. Airy arg from
image-pair delay splitting (geometry.py). NO retrain (post), NO
quad-double (escalate). Housekeeping: gate exact-heavy slow tests. ALL
tests -> Test Developer.

## Build 8e — cusp fast-serving PLANNING START — 2026-07-21
Brief mandates EXACTLY 4 WPs (census, fold/Airy arm, cusp/Pearcey arm,
dispatch), NO housekeeping (tier-split already landed). Code map done:
dispatch = operator.select_branch (RHO_END=4, L_MAX=48), saddle ceiling
_schwinger.W_CEILING_SCHWINGER=60 (_CERTIFICATION_TOL=3e-10). Surrogate
select_chart in surrogate.py, TubeChart.cusp_windows = tuple(theta_cusp,
delta_theta); _tube_serves refuses inside window. Exceptions:
SchwingerCertificationError(RuntimeError), CancellationError(RuntimeError),
LensDomainError(ValueError), HypergeometricDomainError(ValueError). NO new
exception classes. Census script scripts/census_homogenization_corners.py
already classifies schwinger/geometric/refusal geometrically — EXTEND to
add fractions (c) uniform-resolvable and (d) hard-core. Airy arg from
geometry.delay image-pair splitting; Pearcey 2/3 scaling. Refusal-
conservative fall-through; arms are NEW modules + dispatch edits only,
engine internals untouched. All tests -> Test Developer, exact-heavy gated.

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
