---
section: Backlog
---

- **THE SLOW TIERS WERE UNRUN FOR WEEKS AND SURFACED SIX RED FILES — EVERY
  ONE A FIXTURE THAT LEFT ITS SERVED (OR REFUSED) DOMAIN, NOT A CODE
  REGRESSION** `[→ spec]` — measured 2026-08-13 by the driver via
  `.claude/sdk/post_build_sweeps.sh` (`COGWHEEL_BRUTE_ACCURACY=1
  COGWHEEL_TRAIN_TIER=1`, 8-wide over 57 lensing files).

  None was caused by the tier-1 saddle rung: a pytest plugin spying on
  `_saddle_farfield_analytic` measured **calls=0** in every failing file, and
  the three tidied modules are AST-identical to HEAD with docstrings
  stripped.

  ## Closed 2026-08-13

  `test_lensing_ppgo_bandsplit` (4 failed + 4 errors -> **66 passed**) and
  `test_lensing_fold_ppgo_handoff` (2 failed + 2 errors -> **17 passed**).
  See the commit; the fold-ppGO one turned up a structural fact worth
  keeping: the rung first serves near `w ~ 5e4` while the INDEPENDENT
  oracle ceiling is `W_CEILING_SCHWINGER = 60`, so its served domain and
  its verifiable domain are disjoint by **~830x** and "fold-ppGO agrees
  with exact to 1%" can never be tested directly. Same shape as
  [[lensing_saddle_tier1_cannot_reach_the_census_gap]].

  CORRECTED 2026-08-13: an earlier version of this line said the ceiling
  was `W_CEILING_SCHWINGER_QD = 150` (disjoint by ~330x). It is 60.
  `F_op` RETURNS THE UNIFORM ARM for `60 < w <= 150` rather than the exact
  engine, so it is not an independent oracle there — [[FINDINGS F069]].
  And the rung does not merely lack verification: it is measurably WRONG
  by ~21% where it serves, see
  [[lensing_fold_ppgo_rung_serves_wrong]].

  ## OPEN 1 — `test_lensing_airy_fold`, the cusp arm's fixtures

  `_CUSP_FIXTURES` is documented as "fixtures at which the cusp arm SERVES
  (found by a coarse scan)". TWO OF THREE NO LONGER DO. Measured with
  `_capture_cusp_controls` over `w` in 20..500:

      fixture[0] gamma=0.5 r=0.20   never captures controls
      fixture[1] gamma=0.7 r=0.25   never captures controls
      fixture[2] gamma=0.3 r=0.10   captures at every w

  Envelope error vs the exact engine at the ceiling `w = 60`
  (bar `_CUSP_ENVELOPE_TOL = 1e-2`):

      gamma=0.5   1.146e-02   over
      gamma=0.7   1.501e-01   over by 15x
      gamma=0.3   8.381e-03   passes

  Loading the Pearcey table does NOT explain it — with the table set, the
  arm consults the table and `pearcey` is never called, so NO fixture
  captures controls. `cusp_amplification` still returns a value for all
  three (the errors above), so the arm serves by some path while the
  controls path does not fire.

  DO NOT fix this by nudging the fixtures. Two of three fixtures silently
  leaving the arm's serving path, plus a 15x envelope miss at gamma=0.7, is
  the signature of the arm's domain having moved — settle whether that move
  was intended (the `zero_quadrature_pearcey` build killed the live-
  quadrature fallback) before re-pointing any fixture at it.

  ## OPEN 2 — `test_lensing_ratio_layer`, refusal symmetry

  `test_uncertifiable_branch_refused_symmetrically` requires
  `CANCELLATION_CONFIG` (gamma=0.47, y=(0.1,0.1), m_lens=360) to be REFUSED
  by all three paths with the same named exception. All three now SUCCEED.

  This is NOT a lost guard: the three paths agree to **1.93e-2 nats**
  (ratio -575.05928, direct -575.05928, bruteforce -575.03997), inside the
  0.05-nat target, so the engine now certifies what it used to refuse. The
  fixture has left the REFUSAL domain — the mirror image of the usual case.

  Note the constant's own comment records it was already replaced once for
  exactly this ("symmetry premise died. HARD-CORE replacement..."). This is
  the SECOND drift. A third hand-picked config will drift again; either
  derive the witness from the certification boundary at test time, or retire
  the symmetry test and keep the agreement test (which is the stronger
  claim and is what actually held here).

  ## OPEN 3 — `test_lensing_surrogate_census` crown dlnL — ATTRIBUTED

  `LnlTierTestCase::test_real_likelihood_tiers_within_bars`: crown
  dlnL **0.2394** against `CROWN_LNL_TOL = 0.05`.

  ANSWERED 2026-08-13. **No production path returns a wrong likelihood
  here** — the chart is one only the fixture builds. It broke at `4d59a6d`
  (2026-08-07, whose own message says "(build stranded)"), which
  re-coordinatized the fixture box from `s_range`/`d_range` — `d` the
  SIGNED perpendicular distance from the caustic, positive OUTSIDE — to
  `rho_range`/`theta_c_range`, where `rho <= 1` is the INTERIOR. Both the
  semantics AND the axis roles swapped while the numbers were carried
  across verbatim, moving the witness from ~0.125 outside a fold arc to
  `|y| = 0.0271`, essentially the origin.

  Consequence: the locus's `farfield_w_floor` is 352.46 and **100% of the
  served band sits below it**, where `|E_ff| = 272.7` against `|F| = 1.21`.
  The spline is healthy (node-exactness 1.3e-14); ~1e-3 relative error on a
  label 200x larger than `F` is 6.6e-2 of `F` and 0.24 nats. Production
  never builds such a tile — `_farfield_region_w_floor` clips exterior
  tiles — and the fixture reaches it via `from_engine`, which documents
  that it does not guard the exterior contract. Two production defences
  would have caught it: that clip, and the registration bar
  (`farfield_eps_max = 3e-3` vs this chart's 6.6e-2).

  The error is honest gamma-axis interpolation error, not conditioning:
  `carrier_rate` forced to 0 moves dlnL by 2e-5, w-density refinement
  SATURATES (24/dec and 48/dec both ~0.247), and only the gamma axis moves
  it (11 gammas -> dlnL 0.00185, 27x under the bar).

  DO NOT take that gamma refinement as the fix. It is the tempting wrong
  one: it reaches the bar while leaving the chart in a regime production
  would never build, which IS the defect. Nor does simply re-pointing to an
  exterior box work — measured, the constraint is structural: `w_floor >=
  2/max(dtau)` with Fermat delays O(1), so no source position puts a
  60-Msun band above `w_floor` short of `rho ~ 5` where the test goes
  vacuous. The witness LENS MASS has to move too, and the detector band
  spans `f_hi/f_lo = 68.3`, so the whole band must fit between the region
  `w_floor` and the engine ceiling: that works at `rho ~ 1.4-2.0` with
  `m_lens ~ 250-400 Msun`, and fails for `rho <= 1.25` (band top 67-91,
  into the mpmath band).

  So this is a REBUILD of the Section-D fixture, not an edit, and it moves
  all four consumers of `_pos_farfield_dense` (`_likelihoods`, the census
  `run` test, `test_node_exactness`, and
  `test_trough_normalization_stays_bounded`, which needs a genuine `|E|`
  trough the exterior may not provide). Derive the witness FROM the window
  rather than hand-picking it. **Leave the test red until then — it is
  currently the only thing pointing at the serve gap below.**

  ## OPEN 5 — SUB-`w_floor` SERVE GAP: blocks the full-box training campaign

  Found while attributing OPEN 3, and bigger than it. A chart tiled exactly
  as `surrogate_training` tiles it, queried below its `w_floor`, passes
  every gate and serves a value wrong by **468x max|F|** — because
  `_log_w_band_serveable` leaves the low end open and clamps, on a
  justification ("the envelope is smooth and nearly constant below the
  first Airy fringe") that holds for the SACR-C envelope and fails for
  `FARFIELD_KERNEL_SUM`. `farfield_w_floor` appears 0 times in
  `surrogate.py` and `likelihood.py`. Full measurement in [[FINDINGS F070]].

  NOT a today-regression — no `lens_amplification_surrogate.npz` ships, so
  nothing reaches it. It IS a blocker on the deferred full-box training
  campaign: once a chart ships, reachability is generic (any draw whose
  lens mass puts the band bottom below the region `w_floor`, i.e. `m_lens`
  under ~250-800 Msun for these configs). The guard is free at the serve
  site — `_surrogate_coefficients` already holds `geom.delays` and
  `geom.real_mask` — and should mirror the existing `w_trust` split.

  ## OPEN 4 — known, unchanged

  `test_lensing_marginalized_likelihood::test_refusal_precedes_coherent_score`
  — the single entry in `.claude/sdk/known_failures.txt`.

  ## The pattern, and the guard that would have caught it

  Five of six are the same failure mode: a fixture that was inside a served
  or refused domain when written, and is no longer, because a gate moved
  under it. The suites stayed green in the FAST tier and rotted unobserved
  because they are slow-tier gated and the sweeps had not run in weeks.
  `[[lensing_built_but_unused_machinery_guards]]` proposes the cheap greps;
  the analogous one here is a periodic assertion that every named fixture
  still sits where its docstring says it does.

  ## Acceptance

  Re-run `post_build_sweeps.sh` and report per-file. Do not close an item by
  moving a fixture until the gate move that stranded it is understood and
  recorded — a fixture nudge that restores green while hiding a domain
  change is the failure this whole entry documents.
