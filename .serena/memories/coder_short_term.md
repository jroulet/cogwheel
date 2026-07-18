# Coder Short-Term Observations

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
