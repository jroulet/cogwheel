# Inspector Short-Term Observations

## 2026-07-17 — Review of Build 3b (close fast lensed lnlike) uncommitted worktree

Scope: full uncommitted tree in cogwheel-claude-dev. Build 3b was meant to
(WP1) njit-accelerate geometry.nearest_caustic_point's caustic search and
(WP2) fix the kernel node grid for accuracy + land one hook-clean commit.
Reviewed on top of the still-uncommitted Build 3 (numba engine + spline grid).

### Verified GOOD
- WP1 geometry.py: `_caustic_source` reproduces `critical_point`'s `.source`
  arithmetic exactly (macro_matrix inlined correctly: m00=(1-k)-g*cos2b,
  m01=-g*sin2b, m11=(1-k)+g*cos2b; caustic = M@image - image/radius^2). Early
  LensDomainError guard added in nearest_caustic_point (needed because
  `_caustic_source` no longer guards). Final winning-angle `critical_point`
  call preserved, so NearestCausticPoint frame/eigenvalue fields identical.
  `_coarse_squared_distances` njit sweep matches; argsort selection unchanged.
  Value-preserving within ULP. OK.
- WP2 likelihood.py: kernel spline path is legitimate — partition.kernels are
  the SMOOTH K_a(w) with the exp(1j*w*tau_a) delay phase factored OUT
  (reconstructed = sum_a exp(1j w tau_a) K_a), delays kept analytic. Spline
  spans [dense_w.min,dense_w.max] exactly (interpolation, no extrapolation).
  `_full_cluster_delays` virtual delay = delay(caustic.image)-t_min matches
  engine `_labeled_delays` virtual_delay-t_min exactly; real frame matches
  `real_image_delays`. Full-cluster separation placement mirrors F008
  `_channel_switch`. n_kernel_nodes>=4 guard for not-a-knot spline. The
  `[RHO_END/separations.min()]` extra term is redundant (already in
  RHO_END/separations) but harmless. Reduction einsum byte-identical to before.
- operator._contract_orders njit faithfully reproduces the Python order loop
  (two-stage col/row reduction tracks numpy accumulation order; per-element
  index clamp preserved; positive_total companion preserved). F005 refusals all
  stay in Python F_op. (Same as Build-3 review.)

### FINDINGS (build-blocking)
1. ACCURACY DEFECT NOT CLOSED. `_DEFAULT_KERNEL_NODES = 40` FAILS the
   production-grid null-safe interpolation gate:
   test_lensing_fast_path.py::CoarseNodeInterpolationTestCase::
   test_production_grid_reconstructs_below_nullsafe_ceiling
   → two-image null-safe epsilon = 2.758e-2 >> ceiling 1e-3 (n_nodes=44 after
   union). The full-cluster transition placement did NOT rescue two-image (it's
   the slowest-converging config; needs ~82+ nodes). The test's own message
   names the proven-safe fix: raise _DEFAULT_KERNEL_NODES toward ~85 (base=85 →
   8.67e-4 on two-image) — a PRODUCTION change, not a tolerance change. This is
   the SAME class of defect the brief said to fix first (the old default=10).
   RB-vs-brute lnL gate still passed (19 passed), but the interpolation gate is
   part of acceptance and is RED. Suite: 1 failed, 19 passed in 199s.
2. `_DEFAULT_KERNEL_NODES` provenance COMMENT is FALSE: it claims "a base count
   of 40 clears that null-safe target on every _LENS_CONFIGS row" — directly
   contradicted by the failing two-image gate. Same false-provenance pattern the
   brief flagged for the old default=10 comment. Must be corrected together with
   the default.
3. Two Build-3 engine tests STILL RED (unchanged since Build-3 review; _dd.py /
   _hyp1f1.py untouched by 3b):
   - test_lensing_dd.py::SplitterSensitivityTestCase::
     test_broken_splitter_breaks_two_prod — njit froze `_SPLITTER`; py_func
     unwrap still calls compiled `_split`.
   - test_lensing_hyp1f1.py::LadderComplexityTestCase::
     test_shared_numerator_constant_dd_multiplies_linear — monkeypatch can't
     intercept dd_mul/dd_complex_mul inlined into njit _shared_numerator/
     _ladder_sum/_ladder_core (cmul count 0 != 59).
   Brief requires engine suites green at original tolerances → blocks commit.

### Process note
- SPEC.md NOT in the changed set, so the new test module test_lensing_fast_path.py
  is unreflected — the spec/doc pre-commit hook will block the commit
  (build acceptance requires SPEC + fragments). Librarian/commit-step item.

### Bug pattern (carry forward)
njit-inlined primitives break Python monkeypatch/sensitivity tests even with a
`.py_func` outer-unwrap (numba freezes globals at compile & inlines callees).
Recurs whenever a hot primitive gains @njit. Two such tests remain red.
A "measured-safe" default node count taken from a proxy/headline config can
still miss the SLOWEST config (two-image) — gate every config on the PRODUCTION
grid, and set the default from the worst config, not the crown.
