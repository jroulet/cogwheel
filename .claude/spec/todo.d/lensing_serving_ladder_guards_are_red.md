---
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
  [[lensing_fast_tier_hangs_in_mpmath]]) — it wedges around 88%, so its
  summary never prints and the red never surfaces. A gate that cannot finish
  hides ordinary failures as effectively as it hides the hang.

  ## Work

  - Do [[lensing_fast_tier_hangs_in_mpmath]] FIRST — without a completing
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
