# Professor short-term (session 2026-08-06): cusp-adapted wedge axis — INFERENCE REVIEW VERDICT

Reviewed the built tests for the cusp-adapted u=theta^(2/3) wedge axis (SHARD A
test_lensing_interior_wedge_chart.py + SHARD B test_lensing_wedge_dd_arclength.py).
Ran fast tier via /home/tejaswi/anaconda3/envs/cogwheel-newlal/bin/python -m pytest:
**105 passed, 6 skipped, ~114 s.** VERDICT = PASS.

Measured diagnostics (cogwheel/tests/output/), all match spec + first principles:
- T1 transverse cut (gamma=0.3,r=0.455,theta[1e-4,0.2],n_w=10): u max=6.83e-4 (<1e-3 bar),
  p50/p90=1.7e-4/3.7e-4; theta max=8.5e-3; s max=5.48e-2. Ordering u<theta<s holds at
  p50/p90/max. Physics correct: s~theta^2 near cusp -> f(s^1/3) worst; u=d^(2/3) linearizes.
  Oracle = shipping ChangRefsdalChannels engine (independent). Worst-locus reported.
- T2 waist split: gamma=0.5 theta_waist=0.6591 (!=pi/4=0.785, asymmetry real); PHYSICAL
  oracle |r_caustic(gamma,theta_waist)-gamma|<atol asserted (free non-circular value pin,
  as I recommended Q2). axis_origin low/high threaded as tile attr (Q3 honored).
- T3 u-midpoint: theta_split=0.14535 == u_midpoint_image=0.14535, != theta_midpoint=0.2005.
- T4 feedback loop: parent_eps=7.86e-3 gated (>3e-3 bar) -> 4 children eps max=5.82e-4,
  strictly below parent and under bar. Feedback restored.
- T5 node-exact: max residual 5.70e-16 (<1e-14), NPZ v2 bitwise round-trip, schema v2.
- T6: v1 schema hard-refuses with named ValueError.
- SHARD B: per-side closed forms match (LOW theta^2/3-theta_lo^2/3; HIGH sign-flip+offset),
  uniform-in-u grid, wrong-side teeth. SelfFalsification u-map load-bearing: clean p50=2.8e-2
  vs degraded p50=0.36 (13x separation, bar 5e-2).

ONE NOTED DEFERRAL (not a fail): DDWCeilingTestCase (w*r*reach_max<=58 cap binds +
wedge build succeeds under cap) is gated behind COGWHEEL_TRAIN_TIER=1, engine-backed
~85-120 s/node (6 tests ~ 8-12 min) — operator's out-of-band ship gate, correctly outside
my turn budget. Fast-tier coverage of the cap's teeth IS present and green:
NoDDCapLowWTestCase (cap not binding at low w) + SelfFalsification::
test_dd_cap_teeth_uncapped_would_exceed. So cap logic is exercised; only the heavy
positive-binding build is deferred to the operator.
