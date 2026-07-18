# Coder Short-Term Observations

## Build3c WP2: wire channels._exact_total to batched F_op_grid (2026-07-18)

- channels.py `_exact_total`: replaced the per-node scalar `F_op` wave
  loop with a partition. Per-node branch decision (cancellation_exponent
  + select_branch) UNCHANGED; geometric nodes still evaluated inline via
  geometric_amplification (orders=_GEOMETRIC_ORDER, converged=True,
  carrier applied inline). Wave nodes only COLLECTED in the loop
  (wave_indices), then ONE `F_op_grid(w[wave_idx], source, gamma,
  beta=,kappa=,max_order=)` call outside the loop scatters
  values/orders/converged back at their indices. Carrier
  `exp(-1j*w*t_min)` applied to BOTH branches (geometric inline scalar;
  wave elementwise over w_wave). `if wave_indices:` guards the empty
  (all-geometric) case so no empty-array F_op_grid call.
- Import: swapped `F_op` -> `F_op_grid` in the operator import (scalar
  F_op no longer used in channels.py; only docstring mentions remain).
  Updated `_exact_total` docstring (F_op -> F_op_grid, batched note).
- VALUE IDENTITY is structural: scalar F_op and F_op_grid BOTH delegate
  to operator._grid_certified, and _contract_grid processes each node
  independently -> batching cannot change per-node arithmetic. Carrier
  math elementwise-identical to scalar (`-1j*w_wave*t_min` == per-node
  `-1j*float(w[i])*t_min`, same assoc). So byte-identical to pre-WP1
  wave path; speed-only. Caller evaluate() unpacks (exact_total,orders,
  converged) unchanged; operator_orders/operator_converged packaging,
  _physical_kernels, _channel_switch, exact_transition_channels all
  untouched.
- REFUSAL SYMMETRY: a bad wave node raises CancellationError from inside
  the single F_op_grid call (per-node _refusal_message, same threshold
  path) BEFORE any scatter -> propagates unswallowed, identical to old
  per-node raise. LensDomainError (config-level, const in kappa/gamma)
  still raises via geometric_amplification or _grid_certified. No
  catch-and-continue added.

### VERIFIED (server, cogwheel-newlal)
- py_compile OK; `from cogwheel.lensing.chang_refsdal import channels`
  imports (F_op_grid resolves, in operator.__all__). No residual scalar
  `F_op(` call in channels.py (regex F_op\((?!_grid) empty); only
  F_op_grid( calls.

### UNVERIFIED (no accuracy/timing campaign by me; Build3c driver)
- Actual bit-identical lnlike / _amplification_coefficients vs pre-WP1
  across _LENS_CONFIGS (RB-vs-brute <= max(1.5,1e-2|bf|)); reasoned from
  shared _grid_certified path, not run.
- Timing win (fewer numpy dispatches; table/weights built once vs
  per-node) not measured by me.
- Runtime confirmation that a wave-node CancellationError surfaces
  identically on RB and brute paths.

## Build3c WP1: weight-vector batched contraction, single F_op path (2026-07-18)

- operator.py: replaced the per-node 85x85 bilinear `_contract_orders`
  with TWO njit cores (cache=True, fastmath=False):
  `_weight_vectors(...)->(v,v_abs)` scatter-adds the w-INDEPENDENT
  per-order weight vectors v[n,j]=sum_{(a,b):idx==j}
  z[a]*table[n,a,b]*zbar[b] (idx=min(half_sum+n,dim-1)); skips coeff==0
  (== adding 0.0), so clamped (all zero-coeff by half_sum+n<=dim-1)
  entries never scatter into v[dim-1]. `_contract_grid(...)` loops nodes
  x orders x length-dim dot -> per-node (total,positive_total,max_term,
  order_used,last_ratio,converged). ~32M->0.7M contraction flops @105 nodes.
- ONE contraction path via private `_grid_certified(w_array,...)` ->
  (values,orders,converged,estimated_tails,cancellation_ratios): weights
  built ONCE, per-node kernel eval + per-node frexp/ldexp scale, ONE
  `_contract_grid` call, then per-node ALL FOUR F005 refusals
  (non-finite; ratio>1e13; est_tail>1e-10; eps*cond>2e-9) BYTE-UNCHANGED
  + mass-sheet/phase reconstruction. `F_op_grid` = lean
  (values,orders,converged); `F_op` = single-element wrapper rebuilding
  full OperatorDiagnostics (tests read diag.cancellation_ratio/
  estimated_relative_tail). F_op_grid added to __all__.
- Shared internal (not literal "F_op calls F_op_grid") because F_op_grid
  is lean by spec but OperatorDiagnostics needs 2 refusal intermediates
  -> both delegate to _grid_certified. Still ONE contraction, ONE cert.

### VERIFIED (server, cogwheel-newlal)
- py_compile OK. F_op vs F_op_grid agree EXACTLY (diff 0.0) at w=5/8/12,
  y=(0.55,0), kappa=0.05. Refusal intact: y=(0.9,0) gamma=0.2 w=60
  refuses in BOTH grid (one bad node refuses whole grid) and scalar,
  identical msg. No new pyright diags (numba/numpy unresolved pre-existing).

### UNVERIFIED (no accuracy/timing campaign by me; Build3c re-cert job)
- Accum-order change (blocked/dot vs column-then-row) safe in cert band
  per Professor (cond<9e6->pert<2e-8, 200x under 1e-10); needs F005-style
  70-dps mpmath re-cert across L in [24,48] + numba-vs-mpmath gate.
  Reorder perturbs cancellation_ratio << test's 1e-3 vs-oracle tol.
- F010: `_weight_vectors`/`_contract_grid` NEW njit -> self-falsification
  must still go RED via .py_func chain (Test Dev).
- Contraction-flop win not timed by me.


## Build3b WP2: full-cluster node placement + honest default (2026-07-17)

- likelihood.py: fixed the under-resolution accuracy defect. New private
  method `_full_cluster_delays(lens)` mirrors the engine's evaluate()
  delay construction / F008 `_channel_switch` neighbour set: real
  relative delays (`real_image_delays`) UNION the parked virtual-label
  delay at the nearest critical point, in the SAME relative-to-real-min
  frame (`geometry.delay(caustic.image,..) - t_min`). Virtual added ONLY
  when `real_delays.size < _MAX_LENS_IMAGES(=4)` (two-image region); in
  the four-image region it returns real-only -> byte-identical to the
  pre-F008 grid (F008 no-op). Geometry only, no engine sweep;
  LensDomainError raised via macro_matrix/nearest_caustic_point ->
  symmetric with brute path.
- `_coarse_w_node_grid` param renamed real_delays->cluster_delays;
  pairwise RHO_START/sep + RHO_END/sep transition nodes now over
  FULL-CLUSTER separations (real-to-virtual kinks are where two-image's
  binding structure lives). Kept the redundant `[RHO_END/sep.min()]`
  branch node (now full-cluster min; np.unique dedups). NOTE: the real
  branch-switch kink RHO_END/delta_min_real is a real-real separation =>
  a member of full-cluster separations => already among the per-sep
  RHO_END nodes, so F008's real-only `_min_delay_separation` semantics
  are NOT violated by node placement (over-inclusion of break points is
  accuracy-safe).
- `_DEFAULT_KERNEL_NODES` 10 -> 40; deleted the false "F within 1e-8 with
  margin" comment (both the constant comment AND the constructor
  n_kernel_nodes docstring), replaced with honest provenance (null-safe
  max|dF|/max|F|<1e-3; base 40 + full-cluster placement; certified by
  RB-vs-brute + interpolation gates, not the comment).
- Did NOT touch: spline (not-a-knot real/imag), delay analytics,
  LensedBinningError guard, F005/F008 behaviour, kernel_subsamples.
- `geometry` imported from cogwheel.lensing.chang_refsdal (submodule;
  channels.py imports it the same way). Pyright: no NEW diagnostics
  (numpy/scipy unresolved + None-subscript in _build_kernel_subsampling/
  _bin_moments are PRE-EXISTING, env/optional-attr, not mine).

### FLAG for Test Developer (I must not edit tests)
- test_lensing_fast_path.py L59 comment says "default 10-node grid" —
  now 40. And L~830-931 builds a "Refined analogue of production
  `_coarse_w_node_grid`" from `real_image_delays` (REAL-ONLY) — now
  DIVERGENT from production (full-cluster). Per build3b brief this proxy
  must be re-aimed at the PRODUCTION grid with a null-safe metric, and
  SelfFalsification updated so an under-resolved production default
  FAILS. Two-image config is the slowest converger.

### UNVERIFIED (sandbox: no accuracy/timing campaign run by me)
- production RB-vs-brute |lnlike-bf| <= max(1.5,1e-2|bf|) on every
  _LENS_CONFIGS row with base=40 + full-cluster placement, and null-safe
  max|dF|/max|F| < 1e-3 on the PRODUCTION grid per config. If 40 is short
  on two-image, documented fallback is base -> ~82 (convergence table).
  Frame consistency (real relative + virtual relative-to-real-min) and
  separation min-invariance reasoned on paper; needs runtime confirm.

## Build3b WP1: fast caustic search (geometry.py, 2026-07-17)

- geometry.py: added two njit(cache=True, fastmath=False) helpers:
  `_caustic_source(theta, gamma, beta, kappa)` -> source-only point of
  critical_point (skips eigh/eigenframe); `_coarse_squared_distances(
  grid, gamma, beta, kappa, source)` = single compiled sweep for the
  256-pt coarse scan. nearest_caustic_point now: explicit LensDomainError
  guard (byte-identical critical_point message) BEFORE any njit call ->
  njit coarse scan -> 4 bounded minimize_scalar(xatol=1e-12) refinements
  over a Python closure on _caustic_source -> ONE final critical_point
  for the returned frame/eigenvalue. n_grid=256 and 4-cell count kept.
- VERIFIED (server, cogwheel-newlal): _caustic_source vs
  critical_point.source rel 5.4e-15; distance vs refined dense brute
  1.1e-12 and vs old algorithm 1.05e-14 (<1e-10 gate). Branch invariant:
  max ABS virtual-delay diff 3.9e-8 -> w*diff at w=500 = 1.9e-5 vs
  smootherstep width 3.5 (negligible). Image/theta differ up to 8.5e-8
  ONLY at astroid cusps (flat quartic minima) — benign symmetry case,
  same distance/same branch. test_lensing_geometry 11/11,
  test_lensing_channels 21/21 pass.
- TIMING: 20.2ms -> 1.23ms on this box (16x; driver saw ~29ms in engine
  context). Slightly above literal <1ms target but minimize_scalar kept
  per Simplifier; the ~330 eigh critical_point calls (the bottleneck) are
  gone. Remaining ~1.2ms is scipy minimize_scalar Python overhead + final
  eigh. No operator.py/F005 changes.


## WP2 coarse cubic-spline kernel node grid (2026-07-17)

- channels.py: new public `real_image_delays(gamma, y, *, beta, kappa)`
  returning sorted real-image relative (dimensionless) Fermat delays via
  geometry.macro_matrix/find_images/delay ONLY (no F_op). Raises
  LensDomainError symmetrically. Re-exported from chang_refsdal/__init__
  alongside RHO_START/RHO_END (0.5/4.0).
- likelihood.py `_amplification_coefficients`: now evaluates engine ONCE
  at a coarse ~22-node grid (was 506). New `_coarse_w_node_grid(dense_w,
  real_delays)` = np.unique union of geomspace(w_min,w_max,n_kernel_nodes)
  + in-band {RHO_START/Delta_j, RHO_END/Delta_j, RHO_END/Delta_min}.
  CubicSpline(real)/(imag) not-a-knot -> dense_w -> reshape(n_bins,n_sub,
  n_ch) -> SAME einsum k0/k1 reduction (return contract unchanged
  (delays,k0,k1,partition)). New kwarg n_kernel_nodes (default 10, >=4).
- Note: RHO_END/Delta_min is redundant (already in RHO_END/separations)
  but kept explicit per brief; np.unique dedups.

### FLAG for Test Developer (I must not edit tests)
- test_lensing_likelihood.py `_amplification_profile` (~line 397) pairs
  `like._kernel_dense_f` (506) with `partition.exact_total` — partition is
  NOW on the coarse ~N-node grid, so lengths MISMATCH. Rebaseline: pair
  `partition.w`->Hz with `partition.exact_total`, OR reconstruct |F(f)| on
  the dense grid from interpolated kernels + analytic delays (that IS the
  WP <1e-8 reconstruction verification quantity).
- Line ~565 `part.exact_total` used only as scalar max scale — survives,
  value now coarse-grid.

### UNVERIFIED (sandbox: sanity checks only, no accuracy campaign)
- Reconstructed F(f)=sum_a exp(i w tau_a) K_a from spline-interpolated
  kernels within <1e-8 of dense-engine on crown/swept configs (the WP
  accuracy gate) — Test Dev/Inspector to confirm at 1e-8.
- crown RB-vs-brute at RB_ATOL=1.5; F->1 floor 1e-2; warm best-of-5
  lnlike <=10ms on server. Engine-refusal symmetry preserved (no
  try/except added; RB grid != brute grid as before — "symmetric" = both
  propagate unswallowed).

## WP1 numba-njit of Chang-Refsdal per-point engine (2026-07-17)

- Decorated all 17 `_dd.py` scalar dd primitives with
  `@numba.njit(cache=True, fastmath=False)` (arithmetic byte-unchanged;
  fastmath=False is load-bearing — FMA contraction would break the
  two-sum/two-product error-free transforms). njit freezes module-level
  constants (`_SPLITTER`, `_P_*` row indices, `_MIN_ORDER` etc.) at
  compile time — this is numerically fine but BREAKS monkeypatch-based
  tests (see below).
- `_hyp1f1.py`: njit'd `_shared_numerator`, `_ladder_sum`, plus a new
  njit core `_ladder_core(table, carrier, w, max_derivative)` that runs
  the k-ladder. `prefactor_c` (scipy loggamma) + phase-reduction in
  `_carrier` + `_validate_domain` stay in Python; carrier precomputed
  and passed in as `complex(...)`. Q_k ladder prefactor recurred inside
  njit (pure complex arith).
- `operator.py`: extracted the `for order` accumulation loop of `F_op`
  into njit `_contract_orders(...)` returning
  (total, positive_total, max_term, order_used, last_ratio, converged).
  dim=85 matmuls written as explicit two-stage (col-then-row) loops
  mirroring `(z @ M) @ zbar` associativity. ALL FOUR F005 refusal
  checks/thresholds, frexp/ldexp rescaling, mass-sheet reconstruction
  stay in Python and BIT-UNCHANGED.

### Test-side follow-ups (Test Developer owns; Coder must not edit tests)
- `LadderComplexityTestCase::test_shared_numerator_constant_dd_multiplies_linear`
  (test_lensing_hyp1f1.py) monkeypatches `_hyp1f1.dd_mul`/`dd_complex_mul`
  with counting wrappers — under njit the compiled kernels bind the
  compiled primitives, so counts read 0 and asserts fail. Needs
  re-instrumentation via `.py_func` or a redesign.
- `SplitterSensitivityTestCase::test_broken_splitter_makes_two_prod_inexact`
  (test_lensing_dd.py) patches `_dd._SPLITTER` then calls
  `_py(_dd._two_prod)`; two_prod.py_func calls the njit `_split` which
  froze `_SPLITTER` at compile, so the patch may not propagate. Sibling
  `test_broken_splitter_widens_a_half` (patches + `_py(_dd._split)`)
  still works. The `_py` helper docstring already anticipated this.

### UNVERIFIED (sandbox denied execution; runtime-capable roles verify)
- No numba object-mode fallback / successful njit compile of all kernels.
- <=2 ULP agreement vs pre-numba values: explicit-loop summation order
  differs from numpy/BLAS matmul (pairwise). In deep-cancellation configs
  this could exceed 2 ULP in principle; the F005 refusal contract catches
  uncertifiable configs, and certified region has bounded cancellation.
- Green-ness of test_lensing_hyp1f1 / test_lensing_operator oracle suites
  and MacroMagnificationLimitTestCase at original tolerances.
