---
section: Backlog
---
- [ ] **Prior bounds as instantiation arguments; surrogate box =
  coverage, not constraint** `[→ spec]` — OWNER RULING (2026-07-20):
  "in the eventual design, we should use the prior bounds as an
  instantiation argument, with a reasonable default value. if the
  bounds are larger, we can just use geometric optics outside the
  surrogate box." Design consequences to land with / after the 8e
  build and before the full-box training run:
  1. Lens prior classes take their bounds (m_lens range, gamma range,
     source-box scale) as constructor arguments with the current
     values as defaults — no hard-coded coupling of prior support to
     surrogate coverage anywhere.
  2. The serving ladder outside the trained surrogate box is
     explicit and cheap-first: surrogate (in box) -> geometric branch
     (resolved; certified to w <= 500, thresholds per the 8d headroom
     audit) -> 8e uniform-asymptotics patch (near-caustic) -> exact
     quadrature (w <= 60) -> named refusal. Widening the prior NEVER
     requires retraining — it shifts the serving-fraction mix.
  3. The census reports served/geometric/uniform/exact/refused
     fractions AS A FUNCTION of the instantiated bounds, so any
     proposed box widening (e.g. relaxing the w <= 58 mass-
     conditioned source-box shrinkage once 8d+8e land) is decided on
     measured numbers.
  4. High-w physics note (owner, same exchange): the bulk high-w
     regime IS geometric optics — the exact-quadrature arithmetic
     wall (w ~ 64, dd precision) only matters in the near-caustic
     unresolved sliver, which SHRINKS with w and is 8e's mandate;
     the quad-double option stays parked unless the 8d WP3 census
     measures the sliver as non-negligible.
  5. SCALE-RELATIVE TUBE DEPTH (owner design confirmation,
     2026-07-20): node count must stay independent of caustic size —
     the tube interpolates a slowly-varying modulation of the
     universal fold profile, so each adapted axis carries an O(1)
     variation budget PROVIDED the eta band is CAUSTIC-SCALE-RELATIVE
     (eta_max tied to the local curvature radius / reach), not the
     current absolute [0.02, 0.05] (adequate for the 8c fixture
     bands; invalid as gamma -> 0 where the shrinking astroid drops
     below a fixed absolute band — the same foot-of-normal failure
     measured at eta_max = 0.3, size-induced). Far-field boxes
     already reach-scale. Add a tiny-caustic treatment below a gamma
     floor (weak-shear chart in y/gamma-scaled coordinates or the
     analytic limit). Required BEFORE the full-box training run.
     ENFORCEMENT (owner push-back 2026-07-20 — "should have been
     there from the get-go"; make the invariant CHECKED, not
     remembered): the trainer must ASSERT the foot-of-normal
     condition per (band, arc) at build time — eta_max < margin *
     min local caustic curvature radius over the chart domain,
     measured from the caustic geometry it already computes — so
     both the eta_max=0.3 class and the shrinking-caustic class fail
     LOUDLY at training time with no one needing the insight in
     advance. Cost of the late catch was ~zero only because
     TubeChart stores per-chart bounds (computation swap, no schema/
     serving rework) and the production run is sequenced after
     8d/8e; the guard removes the dependence on that luck.
  Links: [[likelihood_cusp-fast-serving]],
  [[likelihood_schwinger-homogenization]],
  [[likelihood_envelope-surrogate]].
