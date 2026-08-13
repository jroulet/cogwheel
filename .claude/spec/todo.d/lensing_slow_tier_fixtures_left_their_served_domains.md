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
  keeping: the rung first serves near `w ~ 5e4` while the exact Schwinger
  oracle ceilings at `W_CEILING_SCHWINGER_QD = 150`, so **its served domain
  and its verifiable domain are disjoint by ~330x** and "fold-ppGO agrees
  with exact to 1%" can never be tested directly. Same shape as
  [[lensing_saddle_tier1_cannot_reach_the_census_gap]].

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

  ## OPEN 3 — `test_lensing_surrogate_census` crown dlnL

  `LnlTierTestCase::test_real_likelihood_tiers_within_bars`: crown
  dlnL **0.2394** against `CROWN_LNL_TOL = 0.05`. Spy-verified NOT the
  tier-1 rung (calls=0). Unattributed — this is the one open failure that
  is a genuine accuracy claim rather than a domain-drift, and it should be
  triaged before the others.

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
