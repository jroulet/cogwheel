# Professor short-term (session: Build 8h-d2 four-defect INFERENCE REVIEW)

Reviewed the four-defect tiling-correctness build (uncommitted worktree; HEAD 2cbab59).
Ran fast domain tests with cogwheel-newlal python. VERDICT: PASS.

## D1 (ppGO exclusion ordering) + D2 (annulus_rho extraction) — GREEN
- test_lensing_ppgo_map.py 22/22, test_lensing_ppgo_bandsplit.py 65/65 (+1 xfail literal bar).
- annulus_rho byte-equivalence exact (matches legacy hypot/reach), input guards name the
  offending arg, zero-magnitude allowed. Matches my Q2 ruling (annulus_rho = |y|/reach).
- Monotonic-conservatism invariant holds: fixed ppGO exclusion reads w_cert <= HEAD (never
  easier); narrowed served region strictly inside outer annulus. Reachable-red guard
  (PpgoOrderingReachableRedTestCase) flips RED on buggy ordering, GREEN on fix — not vacuous.
- Saddle branch byte-identical to HEAD (only positive parity changed). Correct.

## D3 (far-field frame-invariant relabel E_tilde) — GREEN
- test_lensing_farfield_envelope.py 34 passed / 21 skipped (all skips = COGWHEEL_TRAIN_TIER=1
  engine training, operator-deferred). exterior_windows + born also green.
- Telescoping round trip reproduces HEAD to <1e-12 (KERNEL_SUM + MINUS_GHOST); self-falsif
  test_stale_t_min_zero_breaks_the_round_trip confirms exp(+-1jw t_min) round trip is
  load-bearing. Matches my R2.1-R2.4 exact-relabel derivation (E_tilde = absolute-frame
  post-GO remainder = F_abs - sum_a H_a exp(1jw tau_a_abs)).
- Carrier-continuity guard fires on pathological grid, passes continuous; bound = pi/2
  Nyquist quarter-turn (my R2.6). Stale far-field axis-schema hard-refuses at load.

## D4 (cusp-aligned from_engine + positive-box reconstruction) — GREEN
- ClosedFormCuspAngle 4/4: detector matches closed-form {0,+-pi/2,pi} across gamma to <1e-9,
  angles gamma-INDEPENDENT (magnitude varies). CONFIRMS my Q4 closed-form ruling; production
  hardcode validated.
- FromEngineCuspWiring 7/7: positive chart carries cusp node, union dedups/inserts in-range
  cusps, macro-saddle chart has NO cusp nodes (saddle deltoids off-axis, correct);
  test_unpatched_positive_box_build_raises_carrier_discontinuity = reachable-red (fix load-bearing).
- positive-box reconstruction: @expectedFailure REMOVED, POS_RECON_TOL=0.20 NOT widened.
  Cusp union ON -> cusp-ray (gamma=0.40,y1=2.183,y2=0 on theta_c=0 kink) eps~1.1e-4;
  OFF -> 2.6031e-1 (reproduces historical failure). Physics: C2 kink placed ON a spline node.
  CONFIRMED: both positive-box tests PASSED (86s). Kink-on-node fix + E_tilde relabel land it
  within the unchanged 0.20 budget; reachable-red (cusp-union-off regresses to 2.6e-1) green.

## Constraint note
All heavy engine-backed training (COGWHEEL_TRAIN_TIER=1, ~minutes/class) correctly deferred
to the operator's post-build gate; verdict based on fast tests + self-falsification guards +
node-exact identities. No full sampling run touched.
