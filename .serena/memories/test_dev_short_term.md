# Test Dev Short-Term Observations

Build (Born SADDLE census #7, 2026-07-28d): EXTENDED test_lensing_born.py +6
tests (52 total, ~25s green) — SaddleCensusReachableRedTestCase (4) +
SaddleCensusSelfFalsificationTestCase (2). Acceptance #7 saddle 'born' arm of
surrogate_census.classify_fallthrough. Measured facts:
- Witness gamma=1.2,kappa=0: det_a_macro=-0.44<0 (saddle arm), saddle_caustic
  _max_y(1.2,0)=1.618<3 → served; |y|=3.5 in (3,4.2426] → classify='born'.
- Reachable-red for "pre-build positive-only predicate" = mock.patch.object(
  _born,'saddle_caustic_max_y',lambda g,k: math.inf) → saddle branch's
  (max_y<ANNULUS_INNER_RADIUS) always False → falls through 'out-of-box'
  (det<0 fails positive arm's first clause). Faithful because pre-build had NO
  saddle branch.
- |y|=2.0 non-annulus → 'out-of-box' (not born). Grid 5 gammas{1.1,1.2,1.4,
  1.6,1.8}x3 absy{3.05,3.5,4.2}x2 theta = 30 draws ALL 'born', 0 out-of-box
  (theta irrelevant: y2_eig=0 pins |y|, charts=[] so theta never hits a
  chart-serve probe). saddle_caustic_max_y per gamma: 2.08/1.62/1.81/1.98/2.15.
- Self-falsification: gamma=3.0 → saddle_caustic_max_y=3.0>=3 breaches fence →
  NOT born; positive-parity draw (gamma=0.45) served by POSITIVE arm and
  UNAFFECTED by disabling saddle arm (proves arms independent, patch not global
  kill-switch).
- surrogate_census lives at cogwheel/lensing/surrogate_census.py (NOT under
  chang_refsdal/). _GAMMA_GUARD_BAND=1e-3 (in surrogate.py) so gamma=1.2 safe
  from guard. _BORN_ANNULUS_OUTER_RADIUS=3*sqrt2. Reused BornTestCase base,
  _save_plot, CENSUS_BORN_CATEGORY/CENSUS_FALLBACK_CATEGORY/CENSUS_GAMMA(0.45)/
  CENSUS_Y1_EIG(3.6). Added `import itertools`. Plot: saddle_census_tally.png.
- NEIGHBOR now GREEN: test_lensing_surrogate_census.py 14 passed/13 skipped
  (~31s) — the WP3-induced BreakdownPartitionTestCase 16!=14 red flagged in the
  earlier build entry has been FIXED by another run; no longer owed.

Build (Born SADDLE #4/#5/#6, 2026-07-28c): EXTENDED test_lensing_born.py +12
saddle band-split tests (46 total, ~25s green) — SaddleBandSplitCurrencyTestCase
(#4), SaddleGhostRefusedNodeCountTestCase (#5), SaddleLowBandResidualNodeCount
TestCase (#6), SaddleBandSplitSelfFalsificationTestCase. Measured facts:
- #4 witness gamma=1.2,|y|=3.05,theta=0.3: Delta_tau=16.25, r0_sq=212.4 (brief
  said 35.3 — that's |y|=4.24 not 3.05; USE MEASURED). born_gate SERVES at
  w={0.05,0.1} (w*dtau<4) but w*r0>=4 there → r0-currency would refuse
  (reachable-red); REFUSES at w={0.5,1,5}. r0/(2dtau) span across saddle sweep
  =3.9e4x (>>100x).
- #5 gamma=1.6,|y|=4.243,w=5, arc theta[0.02,0.6] (65pt): shipped ppGO-only
  N=2 (|resid|=2.5e-4); ghost-admitted (FARFIELD_KERNEL_SUM_MINUS_GHOST+ghost,
  positive-parity else-branch) N=5, |resid|=7.85e-2 → ~300x inflation. Wider
  arc dips below-split near theta~0.9 → contaminates both variants; keep
  [0.02,0.6]. Wiring: shipped==zero-envelope FARFIELD_KERNEL_SUM to <1e-12.
- #6 gamma{1.1,1.3,1.5} band[1e-3,0.05]: n_logw{5,4,4} n_rad=2 n_az{4,3,3},
  all below-split; gate N<=8. Foil: born_carrier_from_partition(split_constant
  =0.0) forces ppGO on low band → 5.6e5x residual inflation (proves teeth).
- API: ChangRefsdalChannels(wg).reset() returns None (in-place) — do
  ch=...; ch.reset() then ch.evaluate(gamma=,y=,beta=,kappa=). born_gate raises
  _born.BornDomainError. Reused _delta_tau_and_r0_sq/_greedy_node_count/
  _demodulated_residual/_f_exact/_save_plot. Added import functools.
  Plots: saddle_split_currency.png, saddle_ghost_residual.png.


Build (Born SADDLE, 2026-07-28b): EXTENDED test_lensing_born.py +11 saddle
tests (34 total, ~12.5s green) — SaddleCarrierClosedFormTestCase (carrier vs
matrix-solve oracle, Morse -1j), SaddleLeadCarrierF009PinTestCase (|F| flat in
w, phase DOES drift), SaddleExteriorFenceTestCase (band 1.0502342<gamma<3),
SaddleSelfFalsificationTestCase. Key facts:
- Saddle carrier worst rel err = 1.14e-13 at w=8 (phase-amplified round-off in
  phi_geo) — BRIEF's nominal 1e-13 is BELOW float64 reality; gate at 2e-13.
- saddle_caustic_max_y==3.0-to-1e-10 ONLY at the EXACT root
  sqrt((189-15sqrt(105))/32)=1.0502341779 (2.999...9995); brief's literal
  1.0502342 lands 7e-7 off (2.99999929). Use exact root for tight gate,
  literal for serve/refuse straddle (both ±1e-6 straddle since 1e-6>>2.2e-8).
- Fence off/on-axis switch confirmed: at root off=9.0>=on=2.15 (OFF-axis wins);
  at gamma=3 off=7.91,on=9.0 (ON-axis wins) — max_y(3.0)==3.0 EXACTLY (on_axis
  =4*9/4=9). So gamma>=3 refuses; 3.0-1e-6 serves (maxy~3-6.25e-7). gamma just
  above 1 → maxy→inf (u_c→0), so (1,root) all refuse.
- Independent phi_geo oracle = FULL Fermat delay 0.5x0.A.x0 - y.x0 + 0.5y.y
  - ln|x0| (un-collapsed; _born uses A x0=y to drop quadratic). geometry.delay
  is the same formula, genuinely independent of _born.
- Reachable-red verified: sign-flip Morse → carrier gate RED.
ORIGINAL entry: authored test_lensing_born.py (23 tests, ~13s,
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
