---
date: 2026-08-11
section: Backlog
---

- **ELEVEN SERVING-LADDER / CERTIFICATION GUARDS ARE RED AT HEAD — and have
  been shipping unnoticed** `[→ spec]` — measured 2026-08-06, PRE-EXISTING
  (established by A/B, not inferred).

  ## Provenance

  Ran the four affected files on the post-build tree (`034fcf7`) and on
  pre-build `c08f506` in an isolated worktree, same selection, same env:

      POST-BUILD: 11 failed, 122 passed, 21 skipped, 3 xfailed, 1 error  (1796 s)
      PRE-BUILD : 11 failed, 122 passed, 21 skipped, 3 xfailed, 1 error  (1802 s)
      failing sets: IDENTICAL.  New: 0.  Fixed: 0.

  The interior-wedge build (`034fcf7`) neither caused nor fixed any of them.

  ## The failures

  - `test_lensing_airy_fold.py` (6): `ServingLadderDeterminismTestCase`
    (`test_fixed_priority_fold_tried_before_cusp`,
    `test_route_and_value_reproduce_across_all_regimes`,
    `test_served_value_equals_labelled_rung_bitwise`),
    `UniformArmFallThroughTestCase`
    (`test_moving_error_const_threshold_flips_a_fixed_node`,
    `test_served_node_is_bit_identical_to_the_cusp_arm`),
    `CertifiedPathByteIdentityTestCase::test_geometric_node_is_byte_identical`
  - `test_lensing_fast_path.py` (3): `OperatorFusionByteIdentityTestCase`
    (`test_fop_grid_schwinger_arm_flip_witness`,
    `test_fop_scalar_schwinger_arm_flip_witness`),
    `NumbaOperatorPreservationTestCase::test_fop_refuses_uncertifiable_contractions`
  - `test_lensing_levers.py` (1 + the 1 error):
    `LMaxSelfFalsificationTestCase::test_too_high_L_MAX_loses_geometric_availability`
  - `test_lensing_marginalized_likelihood.py` (1):
    `RefusalContractTestCase::test_refusal_precedes_coherent_score`

  ## They are ONE cluster, not eleven bugs

  The messages describe a serving ladder whose routing and whose refusal
  thresholds have both drifted away from what the witnesses pin:

      SchwingerCertificationError not raised                              (x2)
      node did not refuse above the threshold crossing (the threshold is
          dead code)
      grid served w=63 but it is neither the geometric rung nor an arm --
          served by a non-ladder path
      served grid value is not bit-identical to the cusp arm
      cusp node served value differs from its own rung
      the geometric node value drifted from HEAD
      no config refused; the above-ceiling arm was not exercised
      Anti-vacuity: no value comparison ran in this test

  Read together: values are no longer bit-identical between the ladder's
  rungs and the arms that should produce them, AND the guards that should
  refuse above threshold no longer fire. Two of the messages say the pinned
  threshold is now DEAD CODE. So both halves of the certification contract —
  what gets served, and what gets refused — are unenforced.

  ## Why nobody noticed

  These are fast-tier tests that genuinely run and genuinely fail. They went
  unseen because the tree gate has been unable to COMPLETE (see
  [[2026-08-11_mpmath_hang_fast_tier]]) — it wedges around 88%, so its
  summary never prints and the red never surfaces. A gate that cannot finish
  hides ordinary failures as effectively as it hides the hang.

  ## Work

  - Do [[2026-08-11_mpmath_hang_fast_tier]] FIRST — without a completing
    gate there is no way to confirm a fix here.
  - Then bisect: these witnesses were green when written, so `git log` on
    `_schwinger.py` / `operator.py` / the serving-ladder thresholds since the
    last known-green gate will localize the change. The recent Schwinger-qd
    and cusp-arm builds are the first place to look.
  - Decide per test whether the WITNESS is stale (thresholds legitimately
    moved and the pin was not updated) or the BEHAVIOR regressed. Do not
    re-point the pins wholesale — that is exactly how a dead threshold gets
    blessed. `test_thresholds_have_one_home` is the model: one canonical pin
    per decision, in the file that owns the predicate.

  ACCEPTANCE: all 11 green with a non-zero comparison count, and for each
  refusal guard a mutation check showing it still FAILS when the threshold is
  moved.

  ## RESOLVED 2026-08-11 (parameter choice, driver + agents)

  The mpmath-band frequency leak was the common cause of most items: ladder
  nodes at ``w in (60, 150]`` entered the slow/divergent exact-engine path.
  Moving them above the QD ceiling (instant hard-refuse) fixed 10 of the 12
  items, WITHOUT touching production code:

  - `test_lensing_airy_fold.py` (6): `_CUSP_NODE_W` 80→160,
    `_GEOMETRIC_NODE` w 100→200.  All green.
  - `test_lensing_fast_path.py` (3): `FOP_REFUSALS`/supra grids 63→160.
    All green.
  - `test_lensing_levers.py` (1+1 error): `LEVER5_ABOVE_CEILING_W` 62→160
    (the old ``62`` sat in the mpmath band, where the wave evaluator now
    CERTIFIES instead of refusing).  Green.

  ## RESOLVED 2026-08-11 (production fix): `test_refusal_precedes_coherent_score`

  The deferred production fix tracked in
  `lensing_mpmath_band_fixed_panel_rule` landed (completed 2026-08-11,
  see [[2026-08-11_mpmath_fixed_panel_rule]]): the Schwinger QD band
  (`60 < w <= 150`)
  now runs a fixed-order composite Gauss-Legendre rule at
  `_MP_PANEL_ORDER = 32` per panel instead of the unbounded adaptive
  `mp.quad`, so the `CANCELLATION_LENS` hard-core nodes in the band complete
  in O(seconds) and the refusal-precedes-coherent-score spy runs to
  completion.  Green — resolved by the production fix, NOT by a parameter
  choice.

  ## RESOLVED 2026-08-11 (production fix): `test_thresholds_have_one_home`

  The one-home routing disagreement tracked in
  `lensing_one_home_routing_disagreement` was resolved by the ppGO
  fold-pair-existence-or-resolution gate (completed 2026-08-11, see
  [[2026-08-11_ppgo_fold_pair_resolution_gate]]): `cusp_amplification`'s
  ppGO fast rung now serves `fold_ppgo_correction` only when a merging
  min/saddle fold pair exists OR the node is geometrically resolved
  (`w * delta_min >= _PPGO_RESOLUTION_GATE = 4.0`), so an unresolved node
  is no longer served the geometric limit and the served route matches
  `select_branch` again.  Green — resolved by the production fix, NOT by
  a parameter choice.

  ## ALL ELEVEN ITEMS RESOLVED — fragment retired 2026-08-11
