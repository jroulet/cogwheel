# Coder Short-Term Observations

WP2 channels.born_carrier_from_partition saddle branch (Build): added
`is_saddle = (1.0-kappa)**2 - gamma**2 < 0.0` right after gamma/beta/kappa/t_min
reads. Below-split block UNCHANGED (default _born.born_lead_carrier applies Morse
-1j internally -> lead-only serves both parities, one path). Above-split now
branches on is_saddle: POSITIVE arm is the EXISTING code VERBATIM (indented under
else:; farfield_ghost_term caught on geometry.LensDomainError -> envelope*exp(
1j*_frame_phase), reconstruct_farfield w/ FARFIELD_KERNEL_SUM_MINUS_GHOST). SADDLE
arm does NOT call farfield_ghost_term at all (ghost REFUSED: geometry.ghost_kernel
pins sqrt branch w/ reference_amplitude=exp(-0.5j*pi), a POSITIVE-PARITY statement
underived for det A<0, F024; comment names it). Saddle serve: reconstruct_farfield(
w, np.zeros(w.shape,complex), partition.delays, partition.saddle_kernels,
partition.real_mask, FARFIELD_KERNEL_SUM, t_min)[above] -> zero envelope + S_a=1 on
real channels = pure ppGO coherent sum over 2 real images, min-rel frame. `above=
~below` hoisted above branch (same value, shared). ValueError(<2 real) shared,
unchanged. VERIFIED: ast OK, import OK, FARFIELD_KERNEL_SUM exists, reconstruct_
farfield returns (kernels,total). Positive path values byte-identical (else-body
verbatim). UNVERIFIED: no numerical run vs exact saddle ppGO (no test suite per
role). OWED to Test Dev: saddle-partition suite — (1) below-split lead-only both
parities; (2) above-split saddle == FARFIELD_KERNEL_SUM zero-envelope ppGO, no
ghost, finite; (3) farfield_ghost_term NOT invoked on saddle path (spy/patch);
(4) positive-parity byte-identical to HEAD (below+above).

WP3 surrogate_census saddle 'born' arm (Build): added a SECOND born arm in
classify_fallthrough immediately after the positive arm (det_a_macro<0 and
_born.saddle_caustic_max_y(gamma,kappa)<_born.ANNULUS_INNER_RADIUS and
INNER<abs_y<=_BORN_ANNULUS_OUTER_RADIUS -> 'born'). Fence keyed on SHARED
closed-form helper (no inline math; single-source w/ born_gate). saddle_max_y<3
reproduces band 1.0502342<gamma<3 automatically (VERIFIED: g=1.02->4.89 refused,
1.0502342->3.0 boundary, 1.06/1.3/2.9 served). Positive arm VERBATIM (byte-
identical, gamma=0.5 still 'born'). gamma-guard band=0.001 shields wall earlier.
Updated docstring cat-3 note (two arms share annulus, saddle mirrors positive via
shared fence). _born lives at cogwheel/lensing/chang_refsdal/_born.py (imported).
VERIFIED: ast OK, import OK, branch matrix (saddle-annulus->born, inside-inner->
fall-through, positive->born). No test file touched (per WP). OWED to Test Dev:
classify_fallthrough returns 'born' for saddle exterior-annulus draw + mutation
controls (det_a_macro>=0 not saddle-arm, saddle_max_y>=3 refuse, inner/outer edges).

WP1 _born.py saddle carrier (Build: macro-saddle lead carrier + fence):
(a) _born_factors sqrt_mu now 1/sqrt(ABS(det_a)) -> magnitude for BOTH
parities; positive parity bit-identical (abs is identity for det_a>0);
a0/b1 signed formulas unchanged (carry over w/ det_a<0 flipping signs,
F024). (b) born_lead_carrier: det_a=(1-kappa)**2-gamma**2 (beta-indep),
morse=(-1j)**1 if det_a<0 else 1.0 -> return morse*sqrt_mu*exp(1j*w*phi_geo).
EXACT literal -1j (NOT cmath.exp(-1j*pi/2) which injects ~6e-17 real part,
F009-S). pos parity morse=1.0 float -> byte-identical (VERIFIED new==old).
(c) NEW module fn saddle_caustic_max_y(gamma,kappa): F026 closed form,
max(off_axis=4u_c+1/u_c-2, on_axis=4gp^2/(gp+1)) load-bearing cusp switch;
u_c=(sqrt(4gp^2-3)-1)/2. Module self-check assert ==3.0 abs_tol 1e-10 at
gamma=sqrt((189-15sqrt105)/32)=1.0502342 (VERIFIED 2.9999999999999950).
(d) born_gate guard B -> two-sided abs(gamma_p-1)<=DELTA_GAMMA_P (byte-
identical for gamma_p<1; refuses wall strip above 1). Branch: gamma_p<1 keeps
positive fence VERBATIM (raise text unchanged); else saddle_max_y>=3.0 ->
refuse w/ saddle band msg. Guard A (band split w*Delta_tau>=RHO_END via
find_images/delay) SHARED unchanged below branch. Serving band gamma ABOVE
lower root 1.0502342 (extent DECREASES as gamma rises to cusp-switch 1.1777).
(e) born_amplification + born_envelope: fail-loud guard at entry
`if (1-kappa)**2-gamma**2<=0: raise BornDomainError` (abs() removed the
implicit math.sqrt(neg) ValueError these pos-parity diagnostics relied on;
saddle extension out of scope). born_envelope guard placed FIRST (before geom
use) VERIFIED raises w/ geom=None. VERIFIED: ast.parse OK, import+module
asserts OK, gate SERVE/REFUSE matrix (saddle served gamma=1.3|y|=3.4 w=0.01,
saddle-fence refuse gamma=1.04, wall-strip refuse gamma=1.002, pos served
gamma=0.5), |carrier| w-indep to 1e-14 == sqrt(|mu|). UNVERIFIED: no accuracy/
residual-node run vs exact (no test suite per role); channels/census saddle
arms are SEPARATE WPs (WP2/WP3) not touched here. FLAG->Librarian/Inspector:
module docstring WHAT line still says rung serves "at positive parity
(gamma < 3/4)" — now stale (saddle branch added); left unedited (spec/doc
surface ownership).

INS-2-001 FIX (test_lensing_surrogate_census.py): WP3's 6th category 'born'
(_FALLTHROUGH_CATEGORIES = gamma-guard,dropped-sliver,BORN,cusp-window,
refusal-ball,out-of-box) broke BreakdownPartitionTestCase.test_counts_match_
hand_computed (hardcoded 14 -> now 16 via _population looping the tuple).
Fixed category-count-AGNOSTIC: n_cats=len(census._FALLTHROUGH_CATEGORIES),
n_samples=3+1+2*n_cats, served_fraction=3/n_samples (won't rot on future
category adds). Refreshed stale 'five/5-way' wording: module docstring line 10
(added ``born`` to category list, five->six), Section-A comment line 467,
FallthroughCategorizationTestCase docstring 471, BreakdownPartition class
docstring 528. VERIFIED: pytest whole file 14 passed/13 skipped/0 fail.

WP2 channels.py born_carrier_from_partition (Build: band-split carrier assembler):
new fn added AFTER farfield_envelope_from_partition (lines 1293-1462), added to
channels.__all__ and package __init__.py (alongside farfield_envelope_from_partition).
Signature: (partition, *, split_constant=RHO_END(4.0), lead_carrier=None). Split keyed
on invariant w*Delta_tau where Delta_tau = partition.delays[real_mask].max()-min()
(FULL Fermat-delay diff read from partition, NOT phi_geo/w*r0_sq/re-solve, F024).
BELOW (w*Delta_tau<split): _born.born_lead_carrier(w,y1,y2,gamma,beta,kappa) [lazy
import — _born imports channels at module load, circular] evaluated per-w scalar,
demod * exp(-1j*w*t_min) (t_min ONLY from partition.t_min). ABOVE: ghost via
farfield_ghost_term(w,source,matrix,t_min=,real_images=partition.images) tilted
*exp(+1j*_frame_phase) then reconstruct_farfield(..., FARFIELD_KERNEL_SUM_MINUS_GHOST,
t_min) -> F = Sigma_a exp(1j w tau_a) H_a + E (min-rel frame) = ppGO both real images
+ ghost. Ghost tolerance: except geometry.LensDomainError (GhostDomainError IS-A) ->
envelope=0 -> bare ppGO, always finite. Raises ValueError if <2 real images. Passes
raw source+beta (no partial-frame trap — _born rotates internally). VERIFIED: ast.parse
OK, import OK, pkg export OK, born_lead_carrier sig match, MINUS_GHOST in
_FARFIELD_KERNEL_FAMILY (same switch as KERNEL_SUM), reconstruct math (tau_c=0, S_a=1
real). UNVERIFIED: no numerical run vs exact_total (no test suite run per role) — below
frame == exact_total min-rel frame ASSUMED from below/above both landing min-rel; ppGO
leakage-free below ASSUMED by construction (lead-only branch never touches
saddle_kernels). OWED to Test Dev: NEW test_lensing_channels born-carrier suite — (1)
below-split == born_lead_carrier*exp(-i w t_min) node-exact & contains NO 1/w**2 ppGO
inflation as w->0; (2) above-split finite when ghost raises (F023 witness |y|=3.6,
theta=0.5,gamma=0.25,kappa=0.3,beta=0.5); (3) split boundary at w*Delta_tau=4 keyed on
partition-read Delta_tau (mutate real delay -> boundary moves); (4) <2-real-image
ValueError; (5) split_constant/lead_carrier parameterisation (inject stub carrier).


WP1 _born.py (Build: born carrier + band-split): _born_factors now returns
5-tuple (sqrt_mu, phi_geo, q2r, b1, a0) with F023 closed forms
b1=-lam*(2lam r0_sq - x0.y)/(det_a r0_sq), a0=-lam*(lam r0_sq - x0.y)/(det_a r0_sq).
Verified point-mass b1=-1,a0=0 and invariant b1-a0==-lam^2*mu_macro (machine eps).
Added born_lead_carrier (SERVE object, lead-only, no a0/b1). born_amplification/
born_envelope now carry +a0/q2r (resolved-image DIAGNOSTIC only). born_gate: added
gamma<3/4 exterior fence (keyed on caustic max|y| = sqrt(lam)*2gp/sqrt(1-gp) >=
ANNULUS_INNER_RADIUS=3.0, == gamma>=0.75 at kappa=0 -> generalizes correctly for
kappa!=0), re-keyed guard A to band split w*Delta_tau>=RHO_END (RHO_END reused from
operator; Delta_tau = full Fermat-delay span of the two real images via
find_images+delay), guard B unchanged. EPS_BORN demoted to comment (retired T1 bar).
likelihood.py 8h-c1 comment updated (slot still returns None, unwired).
OWED to Test Dev: test_lensing_born.py references _born.EPS_BORN and unpacks the old
4-tuple + pins the b1=1.0 placeholder value -> WILL break at collection; needs porting
(sign fix + a0 + 5-tuple + fence/band-split gates). Serve object is born_lead_carrier.

WP3 surrogate_census 'born' fall-through category (Build): added 'born' as the
3rd slot of _FALLTHROUGH_CATEGORIES (after gamma-guard, dropped-sliver; before
cusp-window/refusal-ball probes, per Professor Q5). New born branch in
classify_fallthrough placed after the dropped-sliver loop / before cusp-window:
returns 'born' when det_A=(1-kappa)^2-gamma^2>0 AND gamma<_born.GAMMA_FENCE(0.75)
AND _born.ANNULUS_INNER_RADIUS(3.0) < hypot(y1_eig,y2_eig) <= _BORN_ANNULUS_
OUTER_RADIUS(3*sqrt2 ~4.2426). |y| from eigenframe coords (rotation-invariant,
no re-solve). Added kappa: float=0.0 keyword-only param (backward-compat; prod
pins 0) for later saddle branch. Sourced inner-radius + fence DRY from
chang_refsdal._born (imported the private submodule; census already reaches
_surrogate privates). Outer radius = box-corner 3*sqrt2 (source box half-width
_Y_SCALE_CAP=3.0). fallthrough_breakdown iterates the tuple dynamically -> auto
-counts 'born'. Verified: parse+import ok, tuple/kappa-default/branch matrix
(annulus->born, inside/outside/fence/saddle->fall-through, kappa-aware).
OWED to Test Dev: test_lensing_surrogate_census.py BreakdownPartitionTestCase.
test_counts_match_hand_computed hardcodes n_samples==14 (2x5 cats) -> now 16
(2x6) via _population() looping _FALLTHROUGH_CATEGORIES; MUST bump to 16. Also
'five-way'/'five categories' docstrings at lines ~10,467,471,528. NEW test owed:
classify_fallthrough returns 'born' for positive-parity annulus draw + mutation
controls (fence gamma>=0.75, saddle det_A<0, inner/outer edges, kappa-aware).
Default _classify fixture uses |y|~0.67 (inside inner edge) so existing
classify tests are unaffected.
