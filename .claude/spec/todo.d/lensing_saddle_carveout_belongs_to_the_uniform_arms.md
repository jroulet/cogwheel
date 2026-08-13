---
section: Backlog
---

- **THE 326 CARVED-OUT SADDLE DRAWS ARE 100% NEAR-CAUSTIC — they are the
  uniform arms' domain, and nothing routes them there** `[→ spec]` —
  measured 2026-08-12 at HEAD 4e72409.

  Causes 2 and 3 of [[lensing_saddle_coverage_gap_breakdown]] are sources that
  fail the lobe admission predicates: 149 inside a lobe failing `admits()`,
  177 inside the charted exterior band failing `admits_exterior()`. Both are
  the `caustic_cloud` nearest-distance `>= eta_max` tube-shell exclusion. The
  open question was whether the carve-out is too aggressive.

  It is not. Their PHYSICAL distance to the caustic:

      inside a lobe    (149)  eta p10 0.008  p50 0.062  p90 0.171
      in exterior band (177)  eta p10 0.020  p50 0.100  p90 0.187

      eta < 0.3 in BOTH populations: 100.0%

  Every one of the 326 sits inside `_ETA_MAX_FOLD = 0.3` — the Airy fold
  arm's certified fence, and the exact complement of `ETA_MIN_GEOMETRIC`. The
  carve-out is doing precisely its job: these are the cusp/fold
  neighbourhoods a polar chart cannot resolve, and the UNIFORM ARMS
  (`_airy_fold`, `_pearcey_cusp`) exist for exactly this region.

  The defect is that nothing routes them there. `_classify_saddle` in
  `census_dry_run.py` returns `exact_engine` and never consults an arm,
  though the census HAS a `cusp_arm` category (11 draws, positive parity
  only).

  ## What must be settled first — the arms' saddle validity

  Do NOT wire this up before checking that the arms are valid for
  `det A < 0`. The fold arm's amplitudes come from fold-frame curvatures and
  look parity-agnostic, but [[lensing_saddle_forensics]] item (f) already
  records that the GHOST branch pin is positive-parity reasoning
  (`geometry.py:2343-2344`, "the real merged saddle ... has Morse index 1"),
  and a wrong branch there is a SIGN error, not a small inaccuracy. The same
  question applies to the arms: measured evidence, not symmetry intuition.

  Also note `_airy_fold`'s fence is PERMANENT and load-bearing: its `xi`
  self-certificate cannot see distance from the caustic (it read 1.2e-2 where
  the true error was O(1) — F028, confirmed against GLoW in F032), and no
  amplitude refinement removes the residual (F033). Any saddle routing must
  respect the same fence rather than widening it.

  ## Structural fact that changes the saddle picture

  `_SADDLE_W_CEILING = 148.0` is set deliberately "2 below
  `W_CEILING_SCHWINGER_QD` (150)". The saddle takes stationary phase only
  when resolved AND `w > 150`. So **the saddle stationary-phase branch is
  UNREACHABLE BY CONSTRUCTION** — the entire saddle domain lives in the wave
  branch (double-double `w <= 60`, mpmath `60 < w <= 148`). This is not an
  artefact of the census sampler, whose `w ~ LogU(5, 148)` mirrors the
  production ceiling exactly.

  Consequence: for the macro saddle there is no "just let the geometric arm
  take it" option at any frequency in the campaign. Every saddle source is
  served by a chart, by an arm, or by the exact engine.

  ## Acceptance

  The `admits`/`admits_exterior` refusal counts (149 + 177) drop by whatever
  an arm-routing change actually claims, reported per-cause against the
  six-way breakdown — not as a change in the total.
