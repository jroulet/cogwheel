# Architect Short-Term Observations

## Build 8g-b — far-field envelope redefinition PLANNING 2026-07-22
Root cause: far-field label = partition.envelope (SACR-C, demodulated at
tau_c w/ criticality switch S_a keyed on w|tau_a-tau_c|); on astroid
diagonals tau_c flips lobes, a resolved image looks near-critical, S_a->0,
its oscillation left UN-subtracted -> env jumps x1500 mid-tile. Fix: for
FAR-FIELD charts ONLY, E_ff = F - sum_{REAL a} H_a e^{i w tau_a} (switch=
real_mask, critical_delay=0, no carrier). Reuse existing gauge:
switched_analytic_channels(w, exact_total, delays, saddle_kernels,
switch=real_mask_float, critical_delay=0.0, weights=_envelope_weights) ->
envelope=E_ff; reconstruct_from_envelope(...same gauge...) inverts to F
exactly (sum u_a=1). ChannelPartition exposes exact_total, delays,
saddle_kernels, real_mask. from_engine (surrogate.py) builds ONLY
far-field charts (tube via _build_tube_chart -> byte-identical). Serve
path: serve() returns E; likelihood _surrogate_coefficients line~1423
reconstruct_from_envelope(geom.switch, geom.critical_delay) -> must
DISPATCH far-field vs tube via a definition tag serve() now returns.
_heldout_eps compares serve envelope to partition.envelope -> far-field
reference MUST switch to E_ff (else gate inconsistent). Tag lives in
FarFieldChart.envelope_definition + _chart_to/from_npz meta; loader hard-
refuses missing/unknown tag (v1/v2 legacy predate tag). Lever3 tiling =
convergence probe (Test Dev) recommends n_per_side 2-3 + y-nodes on new
~1e-4 envelope; gate enforces; driver dials production in v3.

## Build 8g — far-field tiling / eps gate / saddle tail PLAN — 2026-07-22
3 WPs = 3 levers. WP1 eps gate (TrainingConfig tube_eps_max=5e-2,
farfield_eps_max=3e-3; gated OR NaN-eps chart recorded but NOT appended
to charts; must persist eps in per-chart provenance so gate applies on
reuse too). WP2 tiling depends_on WP1 (both rewrite _train_band_charts
far-field section; WP2 must PRESERVE WP1's gate conditional). WP3
saddle-tube-tail independent. Professor: strata per-parity by constant
log-mass factor R=sqrt(51.2)~7.16 -> astroid 3-4 strata, saddle 2 (m>458
beyond-w-cap, recorded loud). DD cap slack; binding = parity ceiling.
Tiling = uniform Cartesian square tiles over [-Y(m_lo),Y(m_lo)]^2, admit
iff box wholly outside caustic_reach+eta_max disk (single 2-image
exterior region, no per-point engine probe); far-field EXTERIOR only.
Reuse PriorBox.y_reach, _capped_w_range, _load_or_build. Tube-tail root
cause = missing wedge-edge guard (wedge walls emit NO cusp_window) +
shallow-cusp under-res; fix SADDLE-ONLY (astroid byte-identical): widen
saddle cusp exclusion + add wedge-edge exclusion window in _saddle_arcs.
All tests -> Test Developer.

## Build 8f — serving micro-levers PLANNING — 2026-07-21
5 levers -> 5 Coder WPs (Simplifier: fold arm-wiring into WP4;
profile-first WPs must commit a benchmark script + pre-identify target).
CRITICAL Professor reframing of Lever 5: RAISING L_MAX is the WRONG
direction — L_MAX is a HANDOFF exponent, not an accuracy floor. The
L~45-46 datum (F005) is the WAVE-branch 1e-10 crossing; geometric-branch
accuracy is governed by w*delta (F013), NOT L (F019-class distinction).
Ship L_MAX=48 UNCHANGED (50 = ceiling of any defensible raise). Retired
the earlier ceil(1.5*L_c) formula. Enforcement = double-sided bracket
L_geo <= L_MAX <= L_wave+margin, both floors MEASURED by Test Dev oracle
sweep at production resolution. Lever5 Coder code = add image-count-match
+ Morse parity-sum (Sum sign(mu_a)==sign(detA)-1) guards to
geometric_amplification (no new exception; reuse LensDomainError) +
corrected provenance on named L_MAX. geometric_amplification confirmed
guard-free today. Lever3 = njit prange pure-map, fastmath OFF, eigenframe
reduce outside loop, any-node-refuses->whole-grid-refuses, byte-identity
==0.0. Lever4 Pearcey box DERIVED at build time (asympt-handoff 1e-8 +15%
margin, caustic 27y^2=-8x^3 inside, stored in provenance), demodulate
Fresnel carrier phi_sp=t*^4+x t*^2+y t*, spline Re/Im separately, 1e-8 abs
on P. All tests -> Test Developer.


## Build 8g triage — 2026-07-22
INS-1-001 (far-field DD product cap drops sqrt(2) corner factor:
_stratum_w_range dd_cap=_DD_PRODUCT_MARGIN/y_max uses per-axis half-width
Y instead of box-CORNER magnitude sqrt(2)*Y, matching prior's _Y_SCALE=307
corner convention and legacy hypot(center)+half*sqrt(2); under-caps by
sqrt(2) so admitted tile corners exceed engine's w*sqrt(s)<=60 ceiling and
get refused -> exactly the coverage holes WP2's tiling exists to close,
though serving stays safe/additive) -> coder_fix. This is a numerical
convention bug in WP2's own new _stratum_w_range helper (introduced by
this build's tiling lever), not a design call Inspector misread — the
sqrt(2) corner-vs-axis distinction is well-established elsewhere in this
codebase (prior's _Y_SCALE, legacy far-field code) so it's an
implementation slip, squarely coder_fix not escalate.

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

## Build 8g-bc triage — 2026-07-22
INS-8gbc-002 (CROWN_LENS relocated fixture sits 2e-4s under DELTA_T_MAX,
0.01966 vs 0.02 -> whole crown/positive fixture family one perturbation
from LensedBinningError; root cause of INS-8gbc-001) -> coder_fix, routed
to Test Developer (test-file-only fixture/constant fix, never Coder,
matches build 8c-cont INS-1-001 precedent). Accepted Inspector's systemic
fix verbatim: raise DELTA_T_MAX and re-derive _shared_fixture fbin so
every far-field-exterior config (incl. kappa=0.1 fall-through, all
LnlikeAccuracy configs) sits <=60% of the new delta_t_max -- physically
sound since fbin spacing is derived FROM delta_t_max (~1/(2*delta_t_max)
phase-accuracy criterion), so raising delta_t_max legitimately produces a
finer, still-valid binning rather than loosening a physical check. Full
re-run + per-config margin report stays with Test Developer since it owns
this fixture; not a Coder self-grading violation because Coder WPs never
touch this file.

## Build 8g-b triage — 2026-07-22
INS-8gb-005 (SPEC.md/DATA_CONTRACTS staleness re farfield_eps_max value,
E_ff redefinition narrative, envelope_definition meta tag) -> override.
Doc-sync is post-gate Librarian/driver responsibility regardless of the
build plan's own prose about "Post-gate doc-sync owns..." — matches prior
INS-1-003 precedent exactly. Flagged forward to Librarian with the two
concrete deltas (farfield_eps_max 3e-3->1e-3; E_ff/envelope_definition-tag
narrative) as exact replacement text.
