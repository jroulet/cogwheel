# Test Dev Short-Term Observations

## 2026-07-17 (PM) — test_lensing_fast_path.py re-aim for Build 3b (WP1+WP2)

- Re-aimed the 5 Architect specs onto the shipped defaults. FINAL:
  19 passed / 1 RED-BY-DESIGN (197 s, threads pinned to 1).
- The single RED gate is
  CoarseNodeInterpolationTestCase::
  test_production_grid_reconstructs_below_nullsafe_ceiling. It is the
  plan-DESIGNED surfacing (build3b_plan ASSUMPTION 1) that the SHIPPED
  _DEFAULT_KERNEL_NODES=40 is under-resolved: two-image null-safe
  epsilon=2.758e-2 >> 1e-3 (4/5 configs short; four-image crown passes
  at 6.7e-5). Convergence sweep: base=40->2.76e-2, base=85->8.67e-4,
  base=128->1.5e-4. Methodology verified faithful to production
  (reconstructed_total, not-a-knot cubic spline, F002 direct-dense
  oracle). Proven-safe fallback = PRODUCTION change (raise
  _DEFAULT_KERNEL_NODES toward ~85) = CODER scope, forbidden for me.
  Did NOT widen the 1e-3 ceiling (forbidden tolerance-dodge) and did
  NOT edit production.
- WHY the lnlike RB-vs-brute gate still PASSES while interp fails:
  Professor's |dlnL|~rho^2*epsilon is PESSIMISTIC; lnlike error is
  RB-binning-dominated, so raw kernel-interp epsilon=2.76e-2 survives
  within the 1.5-nat RB gate. Both gates kept at spec — they measure
  different things.
- SelfFalsification positive control re-pointed from the shipped
  default (now itself short) to CONVERGED_NODES=128 so the falsification
  stays non-vacuous AND green.
- House line-length (79): fixed 5 over-length lines introduced by the
  richer failure messages; AST re-parses; awk clean.

## 2026-07-17 (PM) — Build 3b re-aim of test_lensing_fast_path.py (WP1/WP2)

- Re-aimed the 5 domain gates onto the SHIPPED production node grid.
  Final: 19 passed, 1 FAILED — the 1 failure is BY DESIGN.
- KEY FINDING (plan-anticipated surfacing): shipped
  `_DEFAULT_KERNEL_NODES=40` does NOT clear the blessed null-safe
  interpolation ceiling (1e-3) for 4/5 configs. Driver-confirmed measured
  null-safe epsilon on the PRODUCTION grid: two-image 2.76e-2, near-cusp
  3.7e-3, kappa 3.5e-3, rotated-shear 1.8e-3; ONLY four-image (crown)
  passes at 6.7e-5. WP2's full-cluster transition placement IS wired
  correctly (two-image gets its virtual-label nodes at w=0.44,0.44,3.52,
  3.53) but base=40 log-spacing is too sparse across-band. Convergence:
  base=85 -> 8.67e-4 (clears), base=128 -> 1.5e-4. Fix = raise
  `_DEFAULT_KERNEL_NODES` toward ~85 = PRODUCTION change (Coder scope);
  build3b_plan ASSUMPTION 1 explicitly says "if the gate shows 40 short,
  raise base toward ~82 — surfaced by the gate." So I LEFT THE GATE RED:
  did NOT widen 1e-3 (tolerance dodge forbidden) and did NOT edit
  production. `CoarseNodeInterpolationTestCase::test_production_grid_
  reconstructs_below_nullsafe_ceiling` carries an actionable failure msg.
- IMPORTANT provenance nuance: RB-vs-brute lnlike gate PASSES all 5
  configs at base=40 (d: two 0.066, four 1.19, cusp 0.32, kappa 0.82,
  rot 0.69; all <1.5). So `|dlnL|~rho^2*eps` is PESSIMISTIC here — the
  crown fixture's effective rho is well below the assumed 20, and the
  lnlike error is RB-BINNING-dominated (four-image: eps=6.7e-5 tiny but
  dlnL=1.19 all binning). The conservative 1e-3 gate must still stay to
  protect production rho~20 events where 2.76e-2 WOULD leak ~11 nat.
- SelfFalsification positive control CHANGED: was "shipped default passes
  1e-3" (now false) -> now a CONVERGED base=128 grid (`CONVERGED_NODES`)
  passes, proving the metric can go green; n=4 negative control still
  proves it can go red. Documented deviation; the shipped-default
  shortfall is surfaced by the MAIN gate, not hidden.
- Regression status (NOT mine): geometry+operator 33 passed (WP1 clean).
  dd+hyp1f1 still 2 pre-existing meta failures (SplitterSensitivity,
  LadderComplexity) from the WP1 numba refactor — same as AM note, out of
  scope.


## 2026-07-17 — test_lensing_fast_path.py (WP1 numba + WP2 coarse-node spline)

- Wrote `cogwheel/tests/test_lensing_fast_path.py` (17 tests, all green,
  ~156 s). Covers 4 Architect gates: coarse-node interpolation vs DIRECT
  dense engine (<1e-4, below RB floor — 1e-8 spec was unreachable, all 5
  _LENS_CONFIGS converge 4.4e-6..2.2e-8 at n_base=400); numba kernel/F_op
  accuracy vs INDEPENDENT mpmath (dps=60) + bit-identical repeats +
  domain refusals; few-ms warm timing (lnlike 67.9ms warm, MS_CEILING
  machine-calibrated 0.25s, speedup 163x); crown accuracy anchors.
- ZERO-NOISE FLOOR: RB (coarse-spline) path inherits standard-RB binning
  floor ~8.96e-3 + lensing layer ~2.68e-3 = 1.164e-2. Only the
  BRUTE-FORCE lensed path meets the tight 1e-2 physical floor (residual
  ~1e-11). Gate brute at 1e-2, RB at a separate 1.5e-2 reproduction pin —
  do NOT gate the RB path at the tight floor.
- F_op REFUSALS depend on max_order: FOP_REFUSALS points that refuse at
  the production default MAX_ORDER=42 CONVERGE at max_order=70 (used for
  accuracy). Test the refusal at the DEFAULT order (production operating
  point), not the accuracy order.
- 506 dense sub-sample freqs = n_bins x kernel_subsamples; that is the
  oracle grid for the interpolation gate.
- REGRESSION SEEN (not mine, other suites, flag for Inspector): after
  this build's WP1 numba refactor, 2 meta-tests fail —
  test_lensing_dd.py::SplitterSensitivityTestCase::
  test_broken_splitter_breaks_two_prod ("splitter kept _two_prod exact")
  and test_lensing_hyp1f1.py::LadderComplexityTestCase::
  test_shared_numerator_constant_dd_multiplies_linear. These are
  self-falsification/complexity meta-assertions in OTHER suites (out of
  my scope, not edited). test_lensing_operator.py fully passes.
