# Architect Short-Term Observations

2026-07-17 Build 3b (close Build 3: accurate node grid + hot spots) — IN PROGRESS.
Orientation done. Key code facts:
- likelihood.py:_DEFAULT_KERNEL_NODES=10 (false comment claiming clears 1e-8 gate).
  _coarse_w_node_grid = geomspace(w_min,w_max,n) UNION {RHO_START/Δj, RHO_END/Δj,
  RHO_END/Δmin} in-band. _amplification_coefficients splines coarse kernels (not-a-knot,
  re/im separate) to 506 dense pts (253 bins x 2 subsamples).
- geometry.nearest_caustic_point: 256-pt coarse polar scan + 4x scipy minimize_scalar
  bounded (xatol 1e-12), each calling critical_point (eigh+matmul) → ~330 py calls =29ms/eval,
  w-INDEPENDENT. squared_distance only needs .source (no eigh/frame). Fast path: njit a
  lightweight caustic-source(theta) (no eigh), vectorize coarse scan, keep scipy refine or
  njit bounded minimizer → <=2 ULP, same branch.
- operator._contract_orders ALREADY njit fastmath=False; 2.3ms/pt, order~40 explicit 85x85.
  Floor is FLOP count not JIT. Levers: sparsity skip (bit-safe), series_length/MAX_ORDER
  truncation (accuracy-risky).
- Timing tension: lnlike = caustic + n_nodes*per_pt(2.3ms) + ~4ms. More nodes = accurate
  but SLOWER. few-ms likely UNREACHABLE w/o deferred 2D table → document floor, don't fake.
- test_lensing_fast_path.py: INTERP gate runs on 400-node PROXY (not production);
  SelfFalsification.test_interpolation_gate_rejects_default PROVES default fails. Must re-aim
  at PRODUCTION grid + null-safe metric max|dF|/max|F|. MS_CEILING=0.25 machine-recal (owner dislikes).

DECISIONS (Professor+Simplifier consulted):
- ROOT CAUSE (Professor): two-image slow convergence = _coarse_w_node_grid places transition
  nodes only at REAL-image delay seps Δj, but kernel structure lives at RHO_END/δ_full where
  δ_full = full-cluster sep (incl virtual/parked labels) — SAME real-vs-full-cluster pattern as
  F008. Pure log-spacing needs ~100+ nodes to clear null-safe 1e-3 on two-image (n=82→1.1e-3).
  FIX = place transitions at full-cluster seps (virtual-label delays via geometry.delay at
  nearest_caustic_point critical pt, cheap post-caustic-accel). base=40 + placement clears it.
- WP1 = caustic accel (geometry.py) FIRST: njit _caustic_source(theta) (source only, NO eigh/
  frame), vectorize 256-pt scan, KEEP scipy minimize_scalar 4x refine, ONE final critical_point
  for frame. Accept: dist rel<1e-10 (≤2 ULP target) + IDENTICAL branch across suite. 29ms→<1ms.
  Multiple-minima benign (symmetric, same dist+virtual delay).
- WP2 = node grid (likelihood.py), depends WP1: full-cluster transition placement +
  _DEFAULT_KERNEL_NODES=40 + delete false comment. Interp gate null-safe max|dF|/max|F|<1e-3
  (Professor: |δlnL|~ρ²·ε, ρ~20→0.4nat, 3.7x margin to 1.5).
- DROP contraction WP3 (both: F005 risk, ~20-30% marginal gain, few-ms unreachable regardless).
- few-ms UNREACHABLE w/o deferred 2D surrogate table (per-pt 2.3ms is real FLOP floor).
  lnlike ~110ms @40nodes single-thread — SLOWER than Build3's 70ms but CORRECT. Document floor
  + escalate 2D-table to owner. Do NOT widen tol or fake ceiling.
- NO doc WP (arch rule): has_spec_update=true, doc-sync/Librarian handles SPEC + fragments +
  todo completion + hook. Tests = Test Developer (domain_test_descriptions), not Coder.
</result>
ORIGINAL Build 3 plan (superseded):
2026-07-17 Build 3 (lensed lnlike few-ms) plan:
- Engine call `_amplification_coefficients` = 99.3% of ~20s/eval; per-point ~38ms
  dominated by pure-Python DD 1F1 ladder (`point_mass_g_derivatives`); 506 pts =
  253 bins x kernel_subsamples=2 (kernel grid inherits waveform bin grid).
- Within one eval lens params FIXED, only w=xi*f varies; geometry/delays computed
  ONCE and w-independent. So per-eval work = producing smooth 1D curve K_a(w).
- Professor+Simplifier CONVERGED: Lever1 = numba-JIT the existing DD ladder +
  operator contraction loop (NOT a 2D surrogate table -> deferred Build 3b if
  numba insufficient; table certification is research-grade + F002 risk). Keep
  scipy loggamma / exception-raising / F005 refusal thresholds in Python; njit
  only pure float64/complex128 loops; fastmath=False -> <=2 ULP vs current.
  Lever2 = coarse cubic-spline node grid on K_a(w) (~8-12 nodes, log-spaced UNION
  mandatory transition freqs 0.5/Delta,4.0/Delta + branch 4/Delta_min), splined
  to the 506 bin subsamples; delays stay analytic. FIXED deterministic placement
  (not error-adaptive). K_a is C2 (smootherstep) -> cubic spline matches, Chebyshev
  would ring.
- Gates (Professor authority): numba vs mpmath oracle rel 1e-10 (existing hyp1f1
  suite must stay green, F002 oracle=mpmath); interpolation gate rel 1e-8 on
  reconstructed F(f) vs dense-engine oracle; timing warm best-of-5 lnlike <=10ms
  server (target few-ms); RB-vs-brute RB_ATOL=1.5 crown; F->1 zero-noise 1e-2;
  MacroMagnificationLimit 7.85e-9. NO tolerance widening.
- 2 coder WPs (WP1 numba engine, WP2 coarse grid in likelihood). FINDINGS surrogate
  story -> doc-sync phase w/ measured node count + per-point cost.
