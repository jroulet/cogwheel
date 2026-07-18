# Coder Short-Term Observations

## BUILD 3f INS-5 findings fix — DELIVERED (test suites un-broken, 2026-07-18)

Fixed both Inspector findings by RETIRING obsolete/superseded scaffolding
(both findings explicitly sanction "or retire"). As Coder I did NOT author
replacement certification gates for my own WP1/WP2 code — that circularity is
forbidden. Two test files touched; NO source touched. Both suites now collect
and run GREEN: `pytest test_lensing_channels.py test_lensing_fast_path.py` ->
33 passed in 36 s.

INS-5-001 (test_lensing_fast_path.py, was UNCOLLECTABLE - ImportError on removed
_DEFAULT_KERNEL_NODES):
- Dropped _DEFAULT_KERNEL_NODES from the likelihood import (kept _data_term,
  _norm_term).
- RETIRED CoarseNodeInterpolationTestCase (coarse-node cubic-spline interp gate)
  + its SelfFalsificationTestCase companion method
  test_interpolation_gate_rejects_an_underresolved_grid -> replaced with
  retirement-note comments. KEPT the other 2 self-falsification methods
  (test_kernel_accuracy_gate_rejects_a_perturbed_value,
  test_crown_agreement_gate_rejects_a_shifted_lnl) — both still valid & green.
- Removed orphaned constants (INTERP_NULLSAFE_CEIL, UNDERRESOLVED_NODES,
  CONVERGED_NODES, INTERP_CONFIG_LABELS, INTERP_UNDERRESOLVED_LABELS) and
  orphaned imports (CubicSpline, reconstructed_total). Updated module docstring
  LEVER-2 / INTERPOLATION paragraphs to note the SACR-C retirement.
- GOTCHA caught in this session: the lookahead regex retiring the interp
  self-falsification method left the FOLLOWING method's `def` line at 8-space
  indent (its body was already at 8) -> IndentationError at collection. Fixed
  by re-indenting `def test_crown_agreement_gate_rejects_a_shifted_lnl` to 4
  spaces. LESSON: after a lookahead-anchored method retirement, always re-parse
  — `    def` (4sp) matches as a substring inside `        def` (8sp).

INS-5-002 (test_lensing_channels.py, RealOnlyNeighbourFalsificationTestCase RED
— TypeError: _channel_switch now 4-arg with critical_delay, monkeypatched 3-arg
_real_only_channel_switch):
- RETIRED RealOnlyNeighbourFalsificationTestCase (6 methods) + the 3-arg
  _real_only_channel_switch helper -> retirement-note comments. The F008
  real-only-neighbour rule it falsified is SUPERSEDED by SACR-C's
  criticality-separation switch (report Sec. 6.7).
- SWITCH_REPRODUCTIONS: dropped _real_only_channel_switch -> now
  (_on_caustic_config,). INDEPENDENT_HELPERS shrinks accordingly;
  NonCircularFixtureGuardTestCase iterates it dynamically (asserts
  checked==len(INDEPENDENT_HELPERS)) so no hard reference broke.
- Removed orphaned BUGGY_BLOWUP_FLOOR const + unused `mock` import
  (from unittest import TestCase, main). EPS/_measured_configs still live.
- Updated BoundedKernelTestCase docstring "WHAT IS AND IS NOT COVERED" to record
  that its can-go-red companion was retired.

OWED — TEST DEVELOPER (loud flag, I did NOT author these; they certify WP1/WP2):
1. fast_path: a null-safe INTERPOLATION gate on the SACR-C path — an
   under-seeded LOO-envelope grid must breach the ceiling the adaptive
   _envelope_loo_nodes set clears (replaces the retired coarse-grid gate +
   its self-falsification).
2. channels: a falsification that the SACR-C switch S_a=smootherstep(
   w*|tau_a - tau_c|, 0.5, 4) is load-bearing for BoundedKernelTestCase — e.g.
   inject a 4-arg (w,delays,real_mask,critical_delay) variant keyed on the WRONG
   separation and assert the on-caustic boundedness ceiling goes red. Until it
   lands, BoundedKernelTestCase runs but is not proven able to fail.



## BUILD 3f WP2 — DELIVERED (SACR-C LOO envelope hot path, likelihood.py, 2026-07-18)

Rewired LensedRelativeBinningLikelihood._amplification_coefficients to consume
WP1's SACR-C envelope (reconstruct_from_envelope/_physical_kernels/
_channel_switch from channels.py) instead of the fixed 100-node
_DEFAULT_KERNEL_NODES grid + cubic-spline-of-K_a. ONE file touched
(cogwheel/lensing/likelihood.py); channels.py/_gauge.py byte-unchanged.

Design shipped:
- COARSE EVAL: only the SINGLE smooth envelope E(w) is interpolated; the
  analytic switched saddles S_a*H_a are rebuilt in closed form at every dense
  sub-sample. New _evaluate_envelope(lens, new_w, pad_w): fresh
  ChangRefsdalChannels(grid).evaluate(...) -> (partition, E[keep], Ftot[keep])
  via searchsorted; pads lone nodes to the engine's >=2-point minimum with an
  already-evaluated pad_w (dropped from returns). Geometry fields are
  w-independent (deterministic initial assignment) so disjoint node batches
  stitch consistently.
- LOO REFINEMENT: _envelope_loo_nodes seeds geomspace(w_min,w_max,
  _LOO_SEED_NODES=8) spanning [dense_w.min,dense_w.max]; each iter computes
  _leave_one_out_errors(ln w, E_nodes) (module fn: 4-nearest-OTHER-node cubic
  Lagrange held-out estimate, endpoints=0, O(n), overestimates true global
  spline err => conservative stop), normalizes by max|Ftot| (=max|F|, the
  reconstruction-gate currency; floored at _ENVELOPE_SCALE_FLOOR=1e-12), stops
  when worst/scale < _LOO_STOP=4e-3 (HARD-CODED, not a ctor arg/config key) or
  n>=_LOO_MAX_NODES=48. Else splits the 2 intervals flanking the worst node
  (geometric midpoints), filters near-dupes (isclose rtol 1e-9), clips to
  ceiling room, batch-evals, concat+sort. Self-certifying, config-independent.
- RECONSTRUCTION: _reconstruct_kernels splits E into Re/Im, not-a-knot
  CubicSpline in ln w (REUSES existing complex spline path via real/imag split,
  no new spline machinery), evaluates at ln(dense_w); _physical_kernels +
  _channel_switch closed-form; reconstruct_from_envelope -> kernels(n_dense,4).
  Pure vectorized numpy, NO new njit (so no F010 py_func falsification owed).
  F001 mod-2pi carrier reduction inherited via reconstruct_from_envelope ->
  _gauge._unit_carrier.
- _amplification_coefficients: dense_w, LensedBinningError on non-positive w,
  _envelope_loo_nodes -> (partition,coarse_w,E_nodes); delays = xi*
  partition.delays/(2pi); _reconstruct_kernels; reshape (n_bins,
  kernel_subsamples,4); k0/k1 = same einsum least-squares reduction as before
  (UNCHANGED contraction). Returns (delays,k0,k1,partition); hot path
  (_get_dh_hh_no_asd_drift) still does `delays,k0,k1,_ = ...` (partition
  ignored, retained for API/diagnostics = seed eval, w-indep geometry valid).

REMOVED: _DEFAULT_KERNEL_NODES(=100), _coarse_w_node_grid, _full_cluster_delays,
the n_kernel_nodes ctor arg/attr/validation. (One benign docstring MENTION of
"n_kernel_nodes" remains at ~L469 explaining WHY no such arg exists — not code.)
FROZEN & untouched: _set_summary (stall-ringdown/template builders),
LensedBinningError bin guard, moment contraction (_data_term/_norm_term,
_bin_moments, _build_moment_operators), _build_kernel_subsampling/_kernel_fit_*,
lnlike_bruteforce (exact_total oracle). No tolerance widened.

REFUSAL SYMMETRY: seed grid always includes w_max (worst cancellation,
monotonic in w) so the first engine call raises geometry.LensDomainError
(macro-saddle) / operator.CancellationError (uncertifiable contraction)
unswallowed, matching lnlike_bruteforce — same contract as the retired design.

VERIFIED (read-only static, NOT a test run — Test Dev owns tests): ast.parse OK;
import OK; retired symbols absent; 4 new methods + _leave_one_out_errors + LOO
constants present; __init__ sig has no n_kernel_nodes; _LENS_PARAMS keys
(m_lens_msun,z_lens,y1,y2,gamma,beta,kappa) match _evaluate_envelope/evaluate
usage. UNVERIFIED (downstream, runtime-capable reviewer): RB-vs-brute at
RB_ATOL=1.5 on every regime; near-cusp pin; zero-noise floors; build3f five
gates (recon identity<=1e-13, greedy-oracle N<=26, LOO N<=48, |S_a H_a|<=2,
deep-band F009 constant<1e-6); 18ms warm timing.

FLAGS for Test Developer / Inspector:
- Any test referencing _DEFAULT_KERNEL_NODES / n_kernel_nodes /
  _coarse_w_node_grid / _full_cluster_delays BREAKS (all removed) — rewrite for
  the LOO envelope path (_LOO_SEED_NODES/_LOO_STOP/_LOO_MAX_NODES/
  _ENVELOPE_SCALE_FLOOR).
- New coverage owed: _leave_one_out_errors (held-out estimator; mutation:
  perturb a node, confirm its error rings), _evaluate_envelope (pad_w lone-node
  path + searchsorted keep), _envelope_loo_nodes (stop@4e-3, ceiling clamp,
  dup-filter), _reconstruct_kernels (real/imag spline + closed-form saddles).
- Private imports _channel_switch/_physical_kernels from
  chang_refsdal.channels submodule (package __init__ exports only
  ChangRefsdalChannels/real_image_delays/RHO_START/RHO_END) — sanctioned by the
  reconstruct_from_envelope docstring but a cross-module coupling to note.
- SPEC.md fast-path row still says "_DEFAULT_KERNEL_NODES = 100 ... cubic-splined
  ... K_a(w)"; now describes SACR-C single-envelope interpolation on LOO nodes
  + closed-form saddle reconstruction — Inspector/Librarian sync.



## BUILD 3f WP1 — DELIVERED (SACR-C channels + envelope accessor, 2026-07-18)

Implemented the SACR-C decomposition (envelope_research.md design authority)
in channels.py + _gauge.py. This SUPERSEDES the Build-3e BLOCKED state below:
3f does NOT need the nonexistent per-image wave residual R_j — the residual is
carried by ONE demodulated envelope E, projected onto the 4 physical labels
via per-frequency weights (the telescoping gauge). Only 2 files touched;
operator.py/_hyp1f1.py/geometry.py byte-unchanged (git diff --stat confirms).

_gauge.py (ADDED, existing funcs preserved — tests depend on
exact_transition_channels/unresolved_member_channels/exact_cluster_kernel/
reconstructed_total/smootherstep):
- switched_analytic_channels(w, total, member_delays, saddle_kernels, switch,
  critical_delay, weights) -> (kernels, envelope) [PROJECTION]:
  residual = F - sum_j carrier_j*S_j*H_j; K_j = S_j H_j + alpha_j conj(carrier_j)*
  residual; E = conj(carrier_c)*residual.
- channels_from_envelope(...envelope...) -> (kernels, total) [FORWARD/inverse]:
  residual = carrier_c*E; K_j = trial + alpha_j conj(carrier_j)*residual;
  F = sum_j carrier_j*trial_j + residual.
- envelope_total(w, member_delays, saddle_kernels, switch, critical_delay,
  envelope) -> F  [single authoritative exactness-check reconstruction].
- helpers: _unit_carrier(phase)=exp(1j*(phase - 2pi*round(phase/2pi))) (F001
  mod-2pi reduction, applied ONCE then conjugated so telescoping is machine-
  precision regardless of w*tau); _per_frequency_weights (validate+normalize
  along last axis); _switched_setup (shared validation + carrier/trial/alpha).

channels.py:
- import switched_analytic_channels/channels_from_envelope/envelope_total from
  _gauge; __all__ += 'reconstruct_from_envelope'; const _ENVELOPE_WEIGHT_FLOOR=1e-2
  (= SACR-C eta).
- _channel_switch(w, delays, real_mask, critical_delay): S_a=smootherstep(
  w*|tau_a - tau_c|, RHO_START, RHO_END). This is the F008-SUPERSEDING gate —
  keys on criticality separation |tau_a - tau_c|, NOT full-cluster nearest-nbr.
  Provably >= as conservative for genuine mergers (tau_a->tau_c); accidental
  degeneracies no longer stall. VIRTUAL channels never switch (S=0).
- _envelope_weights(switch)=1 - switch + eta (raw; _gauge normalizes). SINGLE
  home of the weight policy; reconstruct_from_envelope reuses it (DRY).
- evaluate(): critical_delay = virtual_delay - t_min; kernels,envelope =
  switched_analytic_channels(w, exact_total, delays, physical=H_a, switch,
  critical_delay, _envelope_weights(switch)). H_a from geometry.image_kernel
  (CARRIER-FREE per its docstring — correct, carrier applied at reconstruction).
- ChangRefsdalPartition: 7 NEW fields (envelope, saddle_kernels, switch,
  critical_delay, matrix, images, assignment) + property envelope_reconstruction
  (delegates envelope_total). Docstring Attributes updated.
- NEW module fn reconstruct_from_envelope(w, envelope, delays, saddle_kernels,
  switch, critical_delay) -> (K_a, F): the likelihood hot-path forward
  reconstruction; applies _envelope_weights then channels_from_envelope.

4-channel per-freq-weight form CHOSEN over 5th channel (task directive):
keeps _N_CHANNELS=4 so F008 switch neighbour set + label-continuity/crossing
tests unaffected. Documented in module docstring + evaluate() comment.

VERIFIED (minimal smoke invocation, NOT a test campaign — Test Dev owns tests):
gamma=0.2,y=(0.9,0.1),w=geomspace(1,20,40): coherent-sum rel 6.6e-16,
envelope-id rel 1.0e-16, fwd round-trip kernels 6.2e-17 / total 1.0e-16 — all
<< 1e-13 gate. Shapes kernels(40,4)/envelope(40,)/switch(40,4)/saddle(40,4).
F009 deep-unresolved: y=(0.05,0),tiny w -> |F|=1.0206=1/sqrt(1-0.04) exactly,
envelope-id rel 1.3e-23. CancellationError propagates UNSWALLOWED (observed at
w~32, y=(0.9,0.1) — F005 refusal band, correct). Pyright w.ndim/.shape warnings
are false positives matching pre-existing exact_transition_channels/
reconstructed_total style (_as_frequency always returns ndarray).

FLAGS for Test Developer / Inspector:
- F008 switch FORMULA CHANGED (full-cluster -> |tau_a - tau_c|). Any test-side
  reproduction of _channel_switch must align (F002: scenario builders must stay
  channels.py-independent). Crossing/label-continuity behaviour intended
  UNCHANGED (4 labels preserved) but VERIFY green.
- New partition fields + envelope_reconstruction property + public
  reconstruct_from_envelope need coverage (Executable Artifact Verification).
- SPEC.md channel-construction paragraph + DATA_CONTRACTS (if it names channel
  fields) now describe SACR-C — Inspector/Librarian to sync.

---

# (historical, superseded by 3f above)


## BUILD 3e WP2 — BLOCKED (WP1 dependency `transition_envelopes` absent, 2026-07-18)

WP2 ("rewire `_amplification_coefficients` to call
`channels.transition_envelopes(w_coarse)`, interpolate per-image envelopes
R_j, reconstruct dense K_a = S*R_a + (1-S)*(1/alpha_a)*sum_j
alpha_j*R_j*exp(-i*phi_aj)") CANNOT be built: its step-(1) call target
`ChangRefsdalChannels.transition_envelopes` DOES NOT EXIST. Verified this
session: grep `transition_envelopes` across whole tree => hits ONLY in
build3e_plan_approved.json and my own memory; `channels.py` class
ChangRefsdalChannels public API = {__init__, w, reset, evaluate,
evaluate_path} — no envelope method, no `envelope` symbol anywhere in
cogwheel/lensing. WP1 (the method's producer) was BLOCKED (see prior
checkpoint) and never landed. `_amplification_coefficients` in likelihood.py
still holds the Build-3d GLOBAL-spline-over-K_a design (returns
(delays,k0,k1,partition), splines partition.kernels directly) — unchanged.

Root reason WP1/WP2 stall is the SAME unsolved design question, not a coding
gap: the engine produces only the cluster TOTAL F(w) (operator.F_op/F_op_grid)
plus GEOMETRIC per-image kernels (geometry.image_kernel, divergent at caustic,
invalid inside an unresolved cluster). It never computes a per-image
WAVE-OPTICS residual R_j. The plan's "code-pinned" claim that R_j is already
produced by the _dd/_hyp1f1 path (~1us/image/node) is FALSE against the tree
(same finding as WP1). A smooth R_j reproducing exact_total across the
deep-unresolved near-cusp/near-fold band IS the envelope decomposition the
Professor still owes; fabricating it in likelihood.py would be an invented,
unverifiable oracle AND out of WP2 scope (Where: likelihood.py only). Did NOT
touch any file. ESCALATED: WP1 must deliver a real, reviewed
`transition_envelopes` (per-image wave residual) before WP2 can proceed.


## BUILD 3e WP1 — BLOCKED (design premise not code-grounded, 2026-07-18)

WP1 ("thin `transition_envelopes(w)` on ChangRefsdalChannels wrapping
EXISTING `image_amplification_factor(w,j)` -> `_kernel_from_image_amplification`
-> per-image R_j via numba _dd/_hyp1f1, ~1us/image/node") CANNOT be built as
specified: those primitives DO NOT EXIST. Exhaustive search across
cogwheel/lensing: `def image_amplification_factor`, `def _dd_image`,
`def _kernel_from_image_amplification` => zero hits. The approved plan's
"CODE-PINNED efficiency finding" (build3e_plan_approved.json line 21:
`image_amplification_factor -> _dd_image -> _hyp1f1`) is FALSE against the tree.

Architectural reason it's not a trivial add: the engine NEVER computes a
per-image wave-optics amplification F_j. `operator.F_op(w,y,gamma,beta,kappa)`
returns the single CLUSTER TOTAL F(w). Per-image structure exists ONLY as the
GEOMETRIC stationary-phase target `geometry.image_kernel` = H_j = alpha_j*(1 +
iC1/w + C2/w^2), which its own docstring says "is not valid for an image that
is part of an unresolved cluster" (diverges at caustic). `_gauge.exact_cluster_
kernel` is explicit: the cluster kernel = exp(-i w tau)*(total - persistent),
"the divergent per-image asymptotics of the cluster members NEVER APPEAR" — the
design DELIBERATELY avoids per-image cluster amplitudes because they diverge.
`_hyp1f1` public API (point_mass_g_derivatives/_ladder_core/_shared_numerator)
is the operator's derivative-ladder for the TOTAL, not per-image residuals.

Consequence: smooth per-image R_j reproducing exact_total in the UNRESOLVED
band = the unsolved envelope-decomposition (the build3e "design question,
Professor-first"). The only existing per-image split is the artificial gauge
`_member_split` (alpha-weighted demodulated TOTAL) which carries F's beats
verbatim — build3d already showed that's not smooth. A geometric-only R_j =
(1+iC1/w+C2/w^2) is a genuine thin wrapper but reproduces kernels ONLY when
switch=1 (resolved); it FAILS reconstruction in the deep-unresolved near-cusp/
near-fold sub-bands (the binding gate). Building a real per-image wave residual
needs new physics in _hyp1f1.py/operator.py/_gauge.py — all FORBIDDEN by WP1
scope. Did NOT fabricate it (would be an unverifiable invented oracle).
ESCALATED: Professor must produce the ACTUAL decomposition (or the primitive as
a properly-reviewed separate WP) before WP1/WP2 can proceed. No files changed.


## INS-2 ROUND (findings on the global-spline build)

INS-2-003 (likelihood.py speed) — Inspector's suggested remedy (matched-C1
"clamped-to-neighbour-slope" per-segment splines with localized budgets ->
15-20 crown nodes / 15ms) is EMPIRICALLY FALSIFIED. /tmp/probe_segmentation.py
(crown): matched-C1 segmented at C2 landmarks is IDENTICAL to the global
not-a-knot spline at equal node budgets (1.52e-2@20, 5.02e-3@24, 3.52e-3@28,
1.50e-3@36, 3.77e-4@48 — same to displayed precision in both). Reason:
matched-C1 only RESTRICTS the global C2 spline space (drops 2nd-deriv
continuity the true C2 kinks have); it adds no resolution. Pure geomspace(N)
crown dense: N=40->1.8e-3 FAIL, N=56->7.4e-4 PASS — slow ~1/order convergence =
the K_a kernels carry real BEAT oscillation (owner ruling: transition-region
channel construction leaks F's oscillation into nominally-smooth kernels).
=> node count is BEAT-DRIVEN (~50-60 for 1e-3 across configs), NOT a
spline-boundary artifact. 15ms ceiling needs <=~34 nodes (0.395 ms/node
engine); 1e-3 gate needs >=~48-56 nodes. INCOMPATIBLE in-scope. Gap closable
ONLY upstream: Build 3e component-by-component RB (channels.py, OUT OF SCOPE) or
lever-B post-contraction surrogate (chang_refsdal/_surrogate.py, separate Build
3d, OUT OF SCOPE + forbidden dir). No tolerance widened; global spline (holds
HARD accuracy gates) UNCHANGED. Only edit: corrected _coarse_w_node_grid
docstring overclaim ("modest budget suffices") to record the measured beat
floor + matched-C1 falsification + upstream-only reduction.

INS-2-001 / INS-2-002 (test_lensing_fast_path.py) — TEST-DEVELOPER-OWNED
rewrite (I do NOT author tests). Precise spec: (1) line 123 import of removed
_MAX_SEGMENT_NODES/_MIN_SEGMENT_NODES/_SMOOTH_BAND_NODES -> delete (fatal
collection error); reference only the 4 shipped constants
(_LINEAR_NODES_PER_BEAT/_MIN_LINEAR_NODES/_MAX_LINEAR_NODES/
_DEFAULT_KERNEL_NODES). (2) Drop `_segmented_reconstruct` helper (line 265) and
all call sites (740,790,834,854,912,1006) -> use `_global_reconstruct`
(already present). (3) Remove @unittest.expectedFailure at 443,821,843,965
(test_rb_nearfold_exceeds_gate_documented, test_offgrid_reconstruction_gate,
test_dense_reconstruction_gate, test_mutation_rings_at_kinks) — inner
assertions now PASS in shipped code (unexpected-success = failure); assert
green. (4) KEEP @expectedFailure at 653,665,681 (test_crown_node_reduction_
target, test_wellsep_not_heavier_than_crown, test_warm_lnlike_ms_ceiling) —
speed/reduction floor, lever-B, permitted machine-dependent xfail. (5) Rewrite
CoarseNodeGridValidityTestCase: test_at_least_min_nodes_per_segment (528),
test_total_nodes_under_cap (543), test_degenerate_cluster_single_smooth_band
(554) assert per-segment structure (boundaries==[w_min,w_max] only now; no
interior breaks; no _SMOOTH_BAND_NODES fixed count) -> rewrite for the 2-
boundary global scheme. (6) SegmentationNecessityTestCase premise inverted:
production IS global; if kept, assert global<=segmented (12-100x). (7) Replace
inverted node-reduction gate `_DEFAULT_KERNEL_NODES/ncoarse>3` with real metric
vs the 100-node baseline (crown 62 => 1.6x; do NOT assert 4x — falsified).
Module docstring/class names ("per-segment","SEGMENTED grid") + stale narrative
(3.57x,18.8ms,off-grid 3.6e-2) describe removed behavior -> update.

## PRIOR ROUND (INS-1, still in force)

WP1 FINDINGS-FIX (INS-1-001/002/003), cogwheel/lensing/likelihood.py — the
per-segment scheme was FALSIFIED and REPLACED by a single GLOBAL not-a-knot
cubic spline (the empirical winner: mutation tests showed global 12x-100x
better than per-segment at every crown C2 kink).

Root cause of INS-1-002: smootherstep gauge kinks are C2 (only S''' jumps
0->60). A cubic spline is itself C2, so BREAKING the spline at a C2 kink with
fresh not-a-knot BCs on short segments manufactures spurious seam inflections =
strictly worse. Correct move: place a NODE on each kink (as a landmark in the
union), never a segment break.

Current shipped design:
- `_coarse_w_node_grid(dense_w, cluster_delays, delta_min, y_prime_norm)`
  returns (coarse_w, boundaries) where boundaries=[w_min, w_max] ONLY (no
  interior breaks -> the per-segment loop collapses to one global spline).
  coarse_w = unique(concat[ geomspace(wmin,wmax,n_kernel_nodes),
  linspace(wmin,wmax, clip(ceil(_LINEAR_NODES_PER_BEAT*beats),
  _MIN_LINEAR_NODES,_MAX_LINEAR_NODES)), in-band landmarks ]).
  beats=(wmax-wmin)*max_delay_spread/(2pi); landmarks = RHO_START/sep,
  RHO_END/sep over FULL-CLUSTER pairwise seps + branch node
  max(RHO_END/delta_min, L_MAX/y_prime_norm). Log base resolves low-w
  gauge/amplitude; uniform overlay resolves fixed-period high-w delay beats.
- `_amplification_coefficients`: ONE global not-a-knot CubicSpline (real/imag
  separately) over coarse_w -> dense_w; (k0,k1) einsum reduction byte-identical.
- `_full_cluster_delays` returns (cluster_delays, real_delays) [unchanged].
- Constants (locked, gate-robust): _LINEAR_NODES_PER_BEAT=16.0,
  _MIN_LINEAR_NODES=24, _MAX_LINEAR_NODES=56, _DEFAULT_KERNEL_NODES=32.
  Removed: _NODES_PER_SEGMENT_CYCLE,_MIN/_MAX_SEGMENT_NODES,_SMOOTH_BAND_NODES,
  _SEGMENT_MERGE_RTOL. RHO_START/RHO_END/L_MAX imported from operator.

VERIFIED (production path, /tmp/validate_fix.py + worst-case-over-1000-hash-
seeds sweep): dense worst 1.94e-4 (<1e-3, 5.1x); off-grid worst-case-all-seeds
1.98e-4 (<5e-4, 2.5x); RB-vs-brute all <0.09 incl. near-fold 0.051 (gate 1.5).
INS-1-001 correctness/accuracy gates RESTORED green on EVERY config.

INS-1-003 timing (HONEST FLOOR, reported for lever-B escalation — NOT met, no
tolerance widened): warm crown lnlike=27.5 ms at 62 nodes. Profile: engine 1F1
ladder 24.5 ms = 89% (OUT OF SCOPE: operator/_dd/_gauge/geometry); non-engine
overhead only ~3 ms (11%), already minimized by removing per-segment loop.
~0.395 ms/node. Reaching 10ms needs ~18 nodes, which FAILS the HARD 1e-3 gate
(28 already failed). => lever A provably cannot reach 10/15ms while holding the
accuracy gate. Node reduction 100->62 = 1.61x (< 4x). Escalate to lever B
(3D post-contraction surrogate, build3d_brief).

FLAG TO TEST DEVELOPER (test_lensing_fast_path.py):
- Remove @unittest.expectedFailure from test_offgrid_reconstruction_gate,
  test_dense_reconstruction_gate, test_rb_nearfold_exceeds_gate_documented
  (now xpass -> would report FAILURE).
- Keep xfail: test_crown_node_reduction_target, test_warm_lnlike_ms_ceiling
  (speed floor -> lever B; permitted machine-dependent TIMING xfail).
- Line ~123 imports removed constants _MAX_SEGMENT_NODES/_MIN_SEGMENT_NODES/
  _SMOOTH_BAND_NODES -> will break collection; update to the 4 new constants.
- test_global_mutant_not_worse_pinned / test_mutation_rings_at_kinks: production
  IS the global spline now (per-segment removed) — semantics change.
- Remove near-fold exclusion in test_rb_matches_bruteforce_wellconditioned.
- Tests asserting per-segment structure (_MIN_SEGMENT_NODES counts,
  _SMOOTH_BAND_NODES degenerate band, boundaries interior members) need
  rewriting for the global (2-boundary) scheme.

No file under cogwheel/lensing/chang_refsdal/ touched. Refusals
(LensDomainError/CancellationError) still propagate unswallowed on RB+brute.
