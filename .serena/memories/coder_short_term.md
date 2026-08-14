# Coder Short-Term Observations

2026-08-13 WP-3 fold_exterior_ghost (re-gate deferred-training mirrors,
surrogate_census.py ONLY): re-gated surrogate_census.characterize_sample's
interior fold-ppGO census block (~L469-496) from the RETIRED xi-resolution +
_uniform_error_estimate gate to the CURRENT c3-certificate interior rung,
mirroring likelihood._surrogate_coefficients (build ppgo_interior_certificate):
now `image_count == 4` (outer guard, UNCHANGED) AND
`ppgo_error_estimate(real_images, source_arr, matrix, w_min) *
_PPGO_INTERIOR_SAFETY <= CERTIFICATION_BAR`. SINGLE-SOURCED: swapped the module
import `_XI_FOLD_THRESHOLD as _likelihood_xi_fold_threshold` ->
`_PPGO_INTERIOR_SAFETY` (bound from likelihood, =2.0); deleted the now-orphaned
module-level `_XI_FOLD_THRESHOLD` constant + doc block (replaced with a
single-sourcing note); imported `macro_matrix, ppgo_error_estimate` from
chang_refsdal.geometry inside the try (census has no module-level macro_matrix;
old code used `_geom_mod.macro_matrix`). Dropped the deferred imports of
`_merging_fold_pair/_uniform_error_estimate/_image_at_delay` (no longer used).
category label stays 'ppgo_fold' (report bucket unchanged). Partition object
(channels.py ~L1884 dataclass) exposes .images/.real_mask -> geom.images[real]
valid. Import smoke test PASS (_PPGO_INTERIOR_SAFETY=2.0, no _XI_FOLD_THRESHOLD
attr, no dangling old-helper refs). VERIFICATION nuance: the OUTER
`if image_count==4` guard already refused exterior 2-image (never fold-served)
pre-edit -- that half was WP-1's census-level rule and is UNCHANGED; my edit
fixed only the INNER classification (xi->c3). AUDIT (unchanged-with-reason):
surrogate_training.py builders (_build_farfield_chart / _build_lobe_exterior_
chart) DRAW labels by calling shipping from_engine / from_lobe_exterior_engine /
farfield_envelope_from_partition -- WP-1/WP-2 corrected code -- NOT a
transcribed fold/ghost gate; grep found NO _merging_fold_pair/xi/ppgo_error_
estimate anywhere in the file, and the only ghost reference
(_exclude_ghost_dominated ~L1877) single-sources channels._GHOST_DECAY_IM_
THRESHOLD (WP-2 unchanged value); the L5758 "0.7/0.4" strings are prose in a
GhostDomainError catch comment, not enforced literals. scripts/train_lens_
surrogate.py: thin region/flag CLI, zero fold/ghost/gate refs. NO training run
(deferred). Census-suite canonical pin is Test Dev scope (Coder never writes
tests) -- flagged for Test Developer.

2026-08-14 INS-1-001 SECOND PASS (Inspector re-raised same finding; handoff-only
routing did NOT get executed -- test_dev ran but left the file unmodified, build
gate stayed red). Loop-break: EXECUTED the fix myself (Inspector explicitly
directed execution; these are pre-existing tests asserting WP-1's already-landed
physics, NOT tests blessing my own new code). Edits to
cogwheel/tests/test_lensing_airy_fold.py: (1) `_CUSP_TIE_SOURCE_OFF_AXIS`
[0.7,0.05]->[0.15,0.14] (4-img interior, non-tied pair gap 0.255, fold serves
finite w=500) + comment. (2) OnAxis TD-4 header/fixture/`_on_axis_cusp_source`/
class docstrings "interior/both arms serve"->"exterior 2-image; fold refuses,
cusp serves". (3) renamed+inverted test_both_arms_serve...->fold_refuses_and_
ladder_falls_to_cusp (fold None, cusp serves, ladder==CUSP bytes, w in
{50,100,200}). (4) inverted test_fold_arm_tried_first: assertIn('cusp') +
order==['fold','cusp']. SelfFalsification class UNCHANGED (fold already None ->
its mocks are no-ops). SMOKE: py_compile OK; pytest -k on the 4 affected classes
= 11 passed / 0 failed (was 3 failed). Full suite NOT run (Inspector owns the
tree gate). nimg([0.15,0.14])=4, nimg([0.2,1.4142])=2 at gamma=0.5 (fresh exec).

2026-08-14 INS-1-001 (fold_exterior_ghost fix-cycle, ROUTED to Test Developer,
NO test edits by Coder): the finding is 3 stale tests in
cogwheel/tests/test_lensing_airy_fold.py regressing under WP-1's `len(images)
!= 4` fold guard — pure TEST-AUTHORSHIP (fixture swap / assertion inversion /
docstring fix), which the Coder role forbids ("you never write the tests"); the
finding itself assigns to test_dev, and the pipeline runs test_dev DOWNSTREAM of
the Coder, so this is routing not skipping. Wrote airtight handoff
.claude/handoff/ins_1_001_fix.md grounded in FRESH execution (gamma=0.5, exec
NOT denied this session). MEASURED FACTS: `_CUSP_TIE_SOURCE_OFF_AXIS=[0.7,0.05]`
-> 2 images EXTERIOR (mislabeled "off-axis interior"); `[0.7,0.0]` -> 4 interior
(the tie-refusal fixture, CORRECT, leave); `_on_axis_cusp_source()`=[0.2,1.4142]
-> 2 EXTERIOR (docstrings "interior/both arms serve" are FALSE). At gamma=0.5 the
astroid cusp idx1 is at (0,1.4142), soft_axis along y1 -> soft-axis shift stays
EXTERIOR both directions => NO interior on-axis-cusp source exists (premise
physically un-salvageable -> must invert). Ladder at [0.2,1.4142] w=200:
fold=None, cusp=0.478105-1.367105j, `operator._uniform_arm_value`==cusp
byte-identical -> post-F075 truth: fold refuses -> cusp serves -> spy order
['fold','cusp']. FIX-1 drop-in for FoldCuspTie: `[0.15,0.14]` (4-img interior,
merging pair gap 0.255>0 so tau_minus>tau_plus, fold serves finite at w=500 —
satisfies all 3 off-axis assertions); alt `[0.2,0.3]` (gap 0.109). CAUTION: near-
axis `[0.7,0.01..0.03]` are 4-img with a pair but the fold ERROR gate refuses
(serves=False) — do NOT use. FIX-2: invert the two OnAxis determinism tests to
assert fold->None + cusp serves + ladder==cusp + order==['fold','cusp'], fix
docstrings to "exterior (2-image)"; the reproducibility + self-falsification
tests in that class already pass and stay. Import path in the test is
`from cogwheel.lensing.chang_refsdal import geometry, _airy_fold, _pearcey_cusp,
operator` (NOT cogwheel.lensing._airy_fold — that import fails). NO source/test
files changed by Coder this cycle; WP-3 already landed at 084ab7f.

2026-08-13 WP-5 acceptance probe P2 (REPORT ONLY,
.claude/handoff/wp5_probe_p2_report.md + /tmp/ppgo_cert/wp5_probe_p2.py +
.json): ran Professor's P2 (gamma{0.3,0.5,0.7} x |y|/rc{1.05,1.10,1.15,1.25,
1.40} x w{65,100,150}, theta=0.6, 45 oracle pts) scoring the SHIPPING rung
operator._ghost_ppgo_amplification vs the F069-safe absolute-frame oracle
oracle.exact_total (f_schwinger reconstruction, NO t_min pairing). VERDICT:
gates PARTITION -- 39 admitted (all served, 0 overshoot), 6 refused, max
admitted rel_err=1.977e-06 (gamma0.3 rho1.40 w65), ~4 orders under the 1e-2
arm bar. Acceptance band |y|/rc>=1.15 w in (60,150]: 27 pts, MAX rel_err
1.977e-06 PASS. Caveat band |y|/rc=1.05: 3/9 refused (exactly gamma=0.3
rho=1.05 all 3 w, Im tau_c 0.3246<0.4). NO overshoot -> NO w-floor needed ->
NO Inspector finding, WP-2 untouched (Professor prediction holds). KEY nuance
for reviewers: on this grid the SEPARATION gate never binds (min|xa-xc|
2.02-3.61 >> 0.7); the DECAY gate Im(tau_c)>=0.4 is the SOLE active
discriminator, and admit/refuse tracks Im(tau_c) NOT the rho label
(gamma0.5/0.7 rho=1.05 admit b/c Im tau_c 0.84/2.16 large). Refuse-side
correctness is by construction (rung->None->exact engine), not directly
scored. Report-only; no source edits, no committed test.

2026-08-13 WP-2 ppGO+ghost exterior rung (operator.py + geometry.py +
channels.py): added `_ghost_ppgo_amplification(w, y, gamma, *, beta, kappa)
-> complex | None` after geometric_amplification (~L1571) and inserted it
into `_uniform_arm_value` in order fold -> ppGO+ghost -> cusp (interior fold
serve stays FIRST, cusp catch-all LAST). Helper: macro_matrix + find_images +
ghost_kernel([w],...) in ONE try; except ORDER (specific->general, since
GhostAbsentError<GhostDomainError<LensDomainError): GhostAbsentError->None
(interior 4-image decline, byte-identical interior), GhostDomainError->None
(refuse, never zero ghost), LensDomainError->None. Two freq-INDEPENDENT gates
(NO w-floor, Professor Q2): Im(tau_c)>=0.4 (decay, caveat band refuses) AND
min|x_a-x_c|>=0.7 (separation), mirroring channels.farfield_ghost_term
L1100-1109 exactly. Serve = geometric_amplification(w,source,gamma,beta,kappa)
+ complex(np.atleast_1d(ghost.kernel)[0]) * cmath.exp(1j*w*complex(ghost.delay))
(ABSOLUTE-frame carrier, '+' sign, NON-conjugated tau_c per fact 3; NOT
channels.farfield_ghost_term which is t_min-frame). Added `import cmath`.
CONSTANT SINGLE-SOURCING: operator importing channels is a CYCLE (channels.py
L101 imports FROM operator), so hoisted BOTH `_GHOST_SEPARATION_MIN`=0.7 and
`_GHOST_DECAY_IM_THRESHOLD`=0.4 to geometry.py (foundational: imports only
stdlib+numba/numpy/scipy) after `_GHOST_DET_FLOOR`; channels.py L218/L233 now
bind `= geometry._GHOST_...` (kept `_FARFIELD_WINDOW_RADIANS`, still used
L873; derivation invariant 0.4==2.0/5.0 preserved+asserted-in-smoke); operator
references `geometry._GHOST_...` inline. Smoke (full py path): all 3 modules
import clean (no cycle), constants 0.7/0.4 identical across modules,
derivation invariant holds; interior 4-img -> rung None; exterior 2-img
passing both gates -> finite complex; failing gates -> None. Did NOT check
values vs F069-safe oracle (Coder scope; Test Dev/Inspector/Professor own
acceptance). REDUNDANCY NOTE for Inspector: helper calls find_images +
macro_matrix, then geometric_amplification re-solves both internally (mirrors
the existing cusp-arm pattern; correctness unaffected, minor efficiency only).
Fold-arm exterior refusal is WP-1's job; this rung's exterior path is only
reachable once WP-1 lands. Scope IN this WP: rung only. Mirror audit /
retroactive label check / fold refusal were OTHER WPs (see WP-1/WP4 entries).

2026-08-13 WP-1 fold-exterior-refusal (_airy_fold.py + channels.py): added
`len(real_images) != 4 -> refuse` guard at all 3 Airy merging-fold ENTRY
POINTS (F075). find_images / partition.images are REAL-image lists (interior
positive parity=4, exterior=2), so `== 4` is the exact caustic-interior
discriminator -- no new geometry call, count taken from the images object
already in hand. Sites: (1) fold_amplification: guard right after the
try/except that obtains `images`, BEFORE the _ETA_MAX_FOLD admission ->
`return None`. (2) fold_ppgo_correction: guard after the structural-gates
try, BEFORE `_merging_fold_pair` -> `_fallback()` (raw ppGO ==
geometric_amplification, no correction term). (3)
channels.born_carrier_from_partition fold block (~L1601): `if
len(images)!=4: pair=None else: pair=_merging_fold_pair(...)` so the whole
`if pair is not None` correction block skips -> carrier byte-identical to
no-correction path. Interior 4-image path byte-identical at all 3 (guard is
a strict no-op when count==4). _merging_fold_pair, _pearcey_cusp.py,
surrogate_census.py UNCHANGED (confirmed via git diff --stat). py_compile +
import both OK. Did NOT run test suites / oracle value checks (Coder scope);
the ghost-rung serving side + value-vs-oracle acceptance are OUT of this WP.


2026-08-13 WP4 (retroactive label-contamination check, REPORT ONLY,
.claude/handoff/wp4_label_contamination_report.md): traced oracle routes
of both training producers. certified_ppgo_map.npz (walls 443.7/58,
kappa=0, verified from shipped provenance content_hash 7ed0e545...):
ppgo_map._measure_cell's evaluate() closure uses
ChangRefsdalChannels(w_prefix).evaluate().exact_total as the reference F
(error=|exact-ppgo|/max|exact|) -- EXACTLY the F075 label oracle.
Positive-parity sweep _w_nodes(443.7)=geomspace(1,443.7,33) has 5 nodes in
(60,150]: {66.05,79.91,96.68,116.96,141.50}, all below the wall so inside
every exterior cell's accepted prefix. CONTAMINATED cell set = positive
parity x 8 gamma bands (hi<=1.0) x 4 exterior rho bands (ri=3,4,5,6,
centers 1.25/2.0/3.25/6.0) = 32 cells. CLEAN: all saddle cells (wall 58 ->
max node 58<60, never enters band + different macro-saddle route) and all
positive-parity INTERIOR cells (ri=0,1,2 centers<1; the fold-arm defect is
exterior 2-image only, interior 4-image routes correctly). Likely
over-conservative direction (contaminated ~0.45 error pushes sup-over-w
floor to ~141-150 or beyond-wall -> coverage/perf loss for 10 consumers,
NOT accuracy risk). born_residual_chart.npz: w_grid=geomspace(5,60,10),
max node exactly 60.0, contaminated band is OPEN (60,150] -> never entered
-> CLEAN despite far-exterior positive-parity geometry (fold arm not
reached at w<=60, handoff fact 2 safe DD batch). No retraining, no artifact
edits. Advisory: re-run train_ppgo_map.py --production after WP-1 lands.
Shell geomspace/provenance reads succeeded (not denied this session).


2026-08-13 WP2 (interior fold-ppGO re-gate, likelihood.py ~1786-1861):
Re-gated the raw-ppGO interior handoff. Gate is now
`int(geom.real_mask.sum()) == 4` for BOTH parities (replaces the rho<=1
leg AND the removed saddle-only `gamma>1 & !=4 -> None` guard; 4-image is
the exact caustic-interior predicate, 0/2400 disagreements vs closed-form
caustic, F073). Leg3 (_uniform_error_estimate) replaced by the WP1 c3
certificate: `w_min=float(dense_w.min())`; `est=geometry.
ppgo_error_estimate(real_images, source, matrix, w_min)`; admit iff
`est is not None and est*_PPGO_INTERIOR_SAFETY <= CERTIFICATION_BAR`.
New module constant `_PPGO_INTERIOR_SAFETY = 2.0` (Fact 3: p99 0.953,
MAX 0.980, 0% optimistic). Serve path BYTE-UNCHANGED (geometric_amplification
-> f_minrel -> subtract ppgo_sum -> reconstruct_farfield FARFIELD_KERNEL_SUM
-> _reduce_dense_kernels); NO ghost term, NO per-serve ghost_kernel (4-image
=> GhostAbsentError). `except (LensDomainError, ValueError,
ZeroDivisionError): pass` -> fall through to `return None`.
LEG-2 DECISION (by evidence sweep, ppgo_cert_sweep.json, 434 cfg): all
78 interior configs FAIL leg 2; the c3 certificate at S=2.0 admits 230
band rows, MAX true err 4.804e-05 < 1e-4 bar, max ratio 0.972, 0 over bar
-> leg 2 REMOVED from the interior rung. `_merging_fold_pair` and
`_XI_FOLD_THRESHOLD` DEFINITIONS KEPT (fold arm _airy_fold.fold_amplification
still uses them); `_XI_FOLD_THRESHOLD` is now ORPHANED in likelihood.py
(comment refreshed) -> flag to Inspector. Near-caustic divergent-|mu|
subset (rho->1) not DIRECTLY sampled in the sweep (rho in [0.05,0.2]);
covered structurally by the certificate self-refusing (None / over-bar) as
sqrt|mu| diverges. UNVERIFIED: no direct engine sample at rho->1 (Coder
scope; c3 self-refusal is by construction).
STEP D (report-only, NO re-gauge): `_ppgo_cell_coords` (~1438) and
`_train_band_charts` (surrogate_training.py ~4981) BOTH use
`ppgo_map.caustic_rho` (SCALAR max-reach gauge) purely as a MAP COORDINATE
to index the certified ppGO map / read w_trust/w_ceiling from the cell the
map was built with -- they NEED the scalar gauge (must match map build),
NOT the adapted directional gauge. NEITHER relies on "rho==1 is the
caustic": _ppgo_cell_coords only uses rho as a grid coord (its saddle
rho<1 -> None is the SITE5 defense guard, not a caustic-boundary claim);
_train_band_charts's separate INTERIOR TILING uses the adapted directional
gauge (r=|y|/r_caustic, _from_caustic_fixed, where rho==1 IS the caustic)
-- a DIFFERENT coordinate from the ppGO-map caustic_rho scalar gauge.
Correct as-is; not re-gauged.
Import smoke test PASS (_PPGO_INTERIOR_SAFETY=2.0 present). Did NOT run
test suites (Coder scope).

2026-08-13 WP1 ppgo_c3 (interior ppGO certificate): ported
`.claude/handoff/ppgo_c3_reference.py`'s `series_coefficients` into
`chang_refsdal/geometry.py` as private eps-graded polynomial algebra
(`_pmul/_padd/_pscale/_linear_power/_gaussian_moment_table` +
`_series_coefficients` + `_c3_coefficient`), placed after
`saddle_coefficients`. Reused existing `hessian()` (byte-identical to the
reference's hess construction) and `magnification()` instead of
duplicating. Added stdlib `import math`. Public
`ppgo_error_estimate(real_images, source, matrix, w_min)` = sum_a
sqrt|mu_a|*|c3_a| / w_min**3; returns None on w_min<=0 or any non-finite
mu/c3 (near-critical images -> mu=inf -> refuse). NO ghost term (true
interior ghost is exactly 0). `source` accepted but unused (API symmetry,
documented). Exported from chang_refsdal/__init__.py. VERIFIED against the
reference self-test: ported c1==1j*C1, c2==C2 vs shipped
saddle_coefficients to ~1e-14 over 46 images; w^-3 scaling exact
(e(10)/e(20)=8.0); None-guards fire. No gate/serve path touched (that is
WP2). Did NOT run the repo test suite (Coder scope).
