# Build 8e — Cusp fast-serving: millisecond scale everywhere

## Mission

Close the last serving holes so every in-domain query is served at ms
scale (owner mandate: "I would absolutely have a build where it is
millisecond scale everywhere"). Two hole classes exist after 8c+8d:

1. **Cusp exclusion windows** in the 8c tube charts (the (theta_cusp,
   delta_theta) bands where the sqrt(eta) fold model is invalid —
   2/3-power Pearcey scaling governs there).
2. **The unresolved-high-w corner** (w > 60, not geometric-eligible)
   that 8d left as a named `SchwingerCertificationError` refusal
   (F019; measured upper bound ~25% of prior draws carry such nodes).

Owner-ratified design direction (the homogenization seed): the
fold/cusp uniform asymptotics ARE the known structure at the fixed
transition — serve both holes with UNIFORM-ASYMPTOTIC evaluators
(Airy for fold-dominated approach, Pearcey for cusp neighborhoods),
plus smooth interpolated corrections where a bare uniform form misses
the accuracy bars. All uniform machinery belongs to THIS build (the
8d scope fence): nothing was pre-built.

Three concerns:

1. **Corner scoping census (in-build, cheap, FIRST).** 8d measured
   only the dispatch upper bound. Before designing evaluators, measure
   with per-node evaluation over prior draws: of the corner nodes,
   what fraction is (a) geometric under the CURRENT thresholds, (b)
   geometric under thresholds relaxed to the 8d headroom-audit
   evidence, (c) resolvable by the uniform fold/cusp forms
   (w*delta large enough that the asymptotic error term is below the
   bars), (d) genuinely hard (small-w-near-cusp core). The evaluators
   then target (c); (d) is measured and, if non-negligible, ESCALATED
   with numbers (candidate answers: quad-double parked option, prior
   trimming — both owner decisions).
2. **Uniform-asymptotic serving.** Fold/Airy arm for near-fold
   unresolved nodes at any w (the Airy argument is computable from the
   existing geometry: image-pair delay splitting); cusp/Pearcey arm
   for cusp neighborhoods (2/3-scaling; Pearcey function evaluation
   must itself be certified — paired-resolution or oracle-checked,
   never trusted). Serving ladder per node after this build:
   surrogate (in box) -> geometric (resolved) -> uniform (near-fold /
   near-cusp, certified) -> Schwinger exact (w <= 60) -> named refusal
   (only the measured-hard core). Refusal-conservative: any uniform
   evaluation that cannot certify its error bound falls through, never
   serves. NO new exception classes.
3. **Chart-set integration.** The 8c cusp exclusion windows shrink to
   the uniform arms' certified coverage; the census reports the
   fall-through budget before/after (target: fall-through ~0 away
   from the measured-hard core). Chart schema stays; only exclusion
   metadata and the selection ladder change. The 8a/8c
   surrogate-serving contract (never serve where wrong) is untouched.

## Measured facts (pre-answered — do not re-derive)

- 8d corner: ~25% of prior draws carry w > 60 non-geometric nodes
  (dispatch upper bound; Wilson intervals in
  homogenization_corners_census.json). The legacy series is RETIRED
  from production above the ceiling (owner ruling; F019) — do NOT
  resurrect it. Its truncation also failed on part of the corner
  (measured w=100 gamma'=0.2: tail 2.5e-5 > target), so pre-8d
  coverage there was partial anyway.
- Geometric-branch headroom: the 8d audit test (in
  test_lensing_saddle_geometry.py area, Professor Q4 protocol)
  documents current-threshold agreement vs Schwinger at 1e-4 on
  |F|^2 and per-config 1e-4-crossing floors; L > 48 -> L > 60 was
  assessed a likely free relaxation. USE this evidence; any threshold
  change needs the image-count-match guard (missing-image = O(1)
  catastrophe) and 1.5x safety margin.
- The Schwinger ceiling is an arithmetic wall (0.341 digits/unit-w
  against dd's 31.9; certified 60, dead by 68 — measured). Uniform
  asymptotics IMPROVE with w (error ~ inverse powers of the Airy/
  Pearcey argument) — the corner's high-w end is where the uniform
  arms are BEST. The genuinely hard core, if any, is low-margin
  small-argument territory: measure it, do not guess it.
- F017/F018 discipline: theta is gauge (arc-length currency for any
  angular gate); every accuracy claim names its error currency
  (max-normalized envelope error is the lnL-relevant one, F016/F018);
  design-advantage claims are measured against the PRODUCTION
  alternative, not a strawman.
- Test tiers are LAW (CLAUDE.md): in-build tests fast; brute/exact-
  heavy verification gated (`COGWHEEL_BRUTE_ACCURACY`) into the
  driver post-build sweep. The 8d suite re-pricing means any test
  that loops exact evaluations belongs gated from birth.
- HOUSEKEEPING ALREADY LANDED (driver, 2026-07-21 — do NOT re-plan
  it): the exact-heavy tier split is DONE. Gated with the standard
  loud-skip idiom: test_lensing_prior (FoldUnfold/MassSheet brute +
  SamplingSmoke) and test_lensing_marginalized_likelihood (5 heavy
  classes; RefusalContract + BinGuard deliberately KEPT fast as
  load-bearing falsifications). Surrogate training fixtures stay
  ungated BY ADJUDICATION (they underpin fast structural tests; the
  question is deferred to the test-suite curation pass). Plan FOUR
  work packages: census, fold arm, cusp arm, dispatch — no
  housekeeping WP.

## Out of scope — hard fences

- NO surrogate retraining, NO full-box training (rides AFTER this
  build: train once on the final engine + final chart set).
- NO precision-substrate (quad-double) work — owner-parked; if the
  hard-core fraction argues for it, ESCALATE with the measured number.
- NO legacy-series resurrection in production (F019 ruling).
- NO sampling/PP (ruling A). NO enable-by-default changes.
- Engine exact evaluators (_schwinger/_dd/_hyp1f1 internals)
  untouched; the uniform arms are NEW modules plus dispatch-ladder
  edits only.

## Acceptance (two-tier)

1. In-build (FAST): corner-scoping census report (fractions a-d with
   Wilson intervals); uniform arms certified against the exact engine
   on their claimed domains (error bars at the F016-currency bars:
   crown-tier configs <= 0.05 nats impact, strong/saddle <= 0.1 —
   measured through the likelihood on SMALL fixtures); fall-through
   boundaries exercised both directions with the F010 mutation idiom
   (a corrupted uniform evaluation must be refusable and a moved
   boundary must flip decisions); serving-ladder determinism; the
   before/after fall-through budget on fixture charts; suite fast
   tier stays fast (any exact-heavy test born gated).
2. POST-BUILD (driver): full sweep under the flag; corner-core
   escalation memo to the owner if fraction (d) is non-negligible;
   then the full-box training campaign (single run, final engine +
   charts) and the census/price-point package for the enable
   decision.
