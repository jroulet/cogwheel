# Test Dev Short-Term Observations

Build (Born rung, 2026-07-28): authored test_lensing_born.py (23 tests, ~13s,
all green) for WP1 (_born b1-sign/a0/lead-carrier/gamma<3/4 fence/guard-A
re-key) + WP2 (channels.born_carrier_from_partition band-split) + WP3
(surrogate_census 'born' category).
- Coefficient oracle independence: matrix-solve b1 + ANGULAR closed form a0
  (-lam*gamma*cos(2(phi_x0-beta))/det_a), disjoint from _born's algebraic form;
  agree 7e-15 over ~270 combos, gate 2.2e-14.
- a0-inflation ratio is |y|-sensitive: |y|=3.6 gives knife-edge 4.95x (BELOW
  the 5x bar); |y|=3.05 (inner edge) gives 6.45x. Pin the residual-inflation
  test at |y|=3.05, not the brief's mid-annulus radius. Radial y-sweep HIDES
  the a0 break (N_a0=N_lead=2); azimuthal REVEALS it (N_a0=13 vs 3) — the
  method point of F023/F025.
- Split-currency: saddle witness gamma=1.2,|y|=4.2426,theta=0.3 gives
  Delta_tau=35.3; at w=0.05, w*Delta_tau=1.76(<4) vs w*r0_sq=20.6(>=4) →
  opposite split decisions. Positive-parity coincidence (Delta_tau~r0_sq/2)
  only tight at gamma=0.25 (11%); degrades to 2.66x by gamma=0.70.
- NEIGHBOR REGRESSION (WP3-induced, REPORTED not fixed — other owner's suite):
  test_lensing_surrogate_census.py::BreakdownPartitionTestCase::
  test_counts_match_hand_computed FAILS (16!=14). Root cause:
  _FALLTHROUGH_CATEGORIES 5->6 ('born' added); the test loops the tuple for
  per-cat counts but hard-codes n_samples=14 (line 550) and served_fraction
  3/14 (line 559) -> now 16 and 3/16. Fix = 14->16, 3/14->3/16, plus stale
  "five categories"/"2 per category (10)" docstrings (lines 10,467,471,528,547).
  FallthroughCategorizationTestCase (non-annulus fixtures) still passes.
- Diagnostic plots land in cogwheel/tests/output/ (verify via pathlib.glob;
  list_dir is hook-blocked there). _save_plot swallows exceptions so plots
  never fail a gate.
