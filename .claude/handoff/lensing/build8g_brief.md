# Build 8g — Far-field tiling, eps registration gate, saddle-tube tail

## Mission

Make the trained artifact actually cover the prior. The post-campaign
census measured 1/1024 prior draws served: the far-field placement
stage is fixture-scale legacy that builds DUPLICATE boxes at one
hard-coded center, there is no quality gate between a chart's measured
held-out error and registration, and three saddle tube charts carry
0.43-2.15 max-normalized envelope error. Three levers, each
independently certified, additive-serving contract untouched (F005:
the surrogate never overrides a refusal, never serves outside its
validated domain).

1. **Far-field tiling (the coverage lever).** Replace the single-box
   placement in `surrogate_training._train_band_charts` (~line 1147:
   `box_center = (structure.caustic_reach + config.eta_max + 0.2, 0.0)`,
   `half = 0.15`, loop over `max_farfield_regions` builds the SAME box
   under different filenames) with a mass-stratified tiling: partition
   the lens-mass range into log strata so each stratum's whole
   1.7-decade w band fits one chart w range (whole-band containment is
   the serving contract — mirror it, do not change it); per (gamma
   band x stratum), tile the prior's shear-frame y-support box
   (edge `Y(m) = min(307/m, 3)` at the stratum's low-mass edge) with
   DISTINCT boxes, excluding the tube shell (eta <= eta_max) and
   respecting the existing one-image-count-region-per-box constraint.
   Respect parity w-caps via `_capped_w_range`; strata a parity cannot
   reach (saddle beyond w = 58) are recorded loudly in the report as
   beyond-w-cap, never silently dropped. `max_farfield_regions`
   becomes a true cap on distinct tiles with a loud truncation record.
2. **Eps registration gate (the quality lever).** New TrainingConfig
   bar(s) for tube and far-field held-out eps. After `_heldout_eps`,
   a chart above its bar — or with NaN eps (zero held-out points
   served, e.g. the astroid b0/b1 all-refused far-field charts) — is
   recorded in the report with its eps and a `gated` marker and is NOT
   registered into the artifact. Serving in its window falls through
   to the ladder (correct, slower). No serve-time behavior change for
   charts that pass.
3. **Saddle tube tail (the diagnosis lever).** WP-FIRST: diagnose why
   `saddle_b0_tube_5` (eps 2.15, built in 63 s, n_w=32),
   `saddle_b1_tube_2` (1.15, theta window [-0.37, 0.37]) and
   `saddle_b1_tube_5` (0.43) fit so badly while sibling arcs sit at
   1e-2 — plausible candidates: cusp-window adjacency, under-resolved
   theta x w grid, branch/labeling pathology at the deltoid arcs
   tube_2/tube_5 (both recur across bands). Then implement the
   MINIMAL fix (config lever: per-arc node density, window split, or
   wider cusp exclusion) and certify it on a fast synthetic
   reproduction of the worst case, below the bar.

## Measured facts (pre-answered — do not re-derive)

- Census v1 vs the trained full-box artifact (n=1024, seed 0):
  served 1 (0.1%); fall-through: out-of-box 942, dropped-sliver 78,
  gamma-guard 3; engine_refused 0. The one served draw missed exact
  lnL by 2.04 nats (target 0.1).
- Duplicate-box root cause VERIFIED in code and in the training
  report: `farfield_0`/`farfield_1` y_boxes are identical in every
  band (`/home/tejaswi/Work/cogwheel_training/full_box_v1/
  training_report.json`).
- Prior (cogwheel/lensing/prior.py ~lines 66-103): ln m_lens uniform
  on [ln 10, ln 3500]; y = u * Y(m) with u in [-1,1]^2 and
  Y(m) = min(307/m, 3) — the y-box SHRINKS with mass, so high-w draws
  are near-caustic by construction. w(f, m) = 1.2372e-4 * m * f;
  census band 20-1024 Hz -> per-draw w band spans x51.2, sliding
  with m.
- Whole-band containment is the serving contract
  (`surrogate.select_chart` takes log_w_min/log_w_max; census
  `characterize_sample` mirrors production
  `_surrogate_coefficients`).
- Parity w-caps: astroid trained w range [0.0248, 443.7]; saddle
  capped at 58.0 (Schwinger certification wall, F019). Saddle
  whole-band containment therefore tops out near m ~ 458 Msun TODAY.
  The quad-double extension that moves this wall is Build 8h — NOT
  this build.
- Campaign artifact: 86 charts, 54.3 MB. Far-field chart: 600 engine
  calls, ~350 KB, 7-610 s build. Tube chart: 1344 calls, ~760 KB,
  63-278 s. Bands: astroid 9, saddle 7; 7 dropped gamma-slivers
  (~7.6% of draws, by design, recorded).
- Chart quality (max-normalized envelope currency, `_heldout_eps`):
  far-field median 3.7e-4 (n=32); tube median 3.8e-2 (n=54); tail
  2.15 / 1.15 / 0.43 plus five more >= 0.09, all saddle, arcs
  tube_2/tube_5 recurring.
- Dead charts (SAFE but dead weight): astroid b0/b1 far-field,
  600/600 training points engine-refused -> NaN eps; exclusion balls
  cover the whole box so they can never serve. Tiny-gamma
  (gamma <= 0.06) additionally has all 6 tube arcs foot-of-normal
  skipped: no coverage there, ladder serves it.
- Test tiers are LAW; the tree-wide fast gate is a commit
  precondition (SDK preflight). In-build training runs are small
  synthetic configurations ONLY.

## Program north star (owner ruling 2026-07-22 — binds the design)

The FINAL serving result never falls through to full evaluation /
quadrature: every prior draw is served by the surrogate OR by
post-post-geometric-optics (ppGO / uniform arms, which at high impact
parameter are essentially exact). Exact quadrature (Schwinger) is a
TRAINING-LABEL and VALIDATION tool only. For this build that means:
(1) the tiling design targets full coverage of the region ppGO does
not certify — not merely "where the ladder is slow"; (2) fall-through
to exact after a gated/missing chart is acceptable MID-PROGRAM but is
a defect in the final artifact — gated windows get rebuilt, not
abandoned; (3) the build report must MEASURE the prior fraction
served by neither charts nor certified ppGO (the residue that Build
8h labels + arms must close). Zero-exact-serving is the program's
final census acceptance.

## Out of scope — hard fences

- NO Schwinger/engine numerics changes; NO quad-double work (Build
  8h, separate owner-approved track).
- NO serving-contract changes: whole-band containment, the guard
  stack, exclusion balls, and the refusal vocabulary stay exactly as
  they are. NO per-node/segment serving.
- NO full-box campaign in-build — that is the driver's post-build
  step (charts are resumable; astroid tubes/far-fields will be
  reused or superseded by the driver, not by agents).
- NO census module rewrite (a small serve-fraction smoke helper as a
  test utility is fine).

## Acceptance (two-tier)

1. In-build (FAST, synthetic-scale):
   (a) a small synthetic training run produces >= 3 DISTINCT
   non-overlapping far-field tiles covering a specified y-annulus,
   every tile recorded in the report, any cap truncation and any
   beyond-w-cap stratum recorded loudly;
   (b) serve-fraction smoke on that synthetic artifact: draws inside
   the tiled support serve at >= 90%; draws outside still fall
   through (additive-only contract exercised both directions);
   (c) eps gate: a deliberately poisoned chart (mutated labels) and a
   NaN-eps chart are both excluded from registration and recorded
   (F010 reachable-red on the gate);
   (d) tube tail: a written diagnosis of the pathological arcs plus
   the chosen minimal fix rebuilding the worst-case synthetic
   reproduction below the bar;
   (e) fast tier green (tree-gate preflight).
2. POST-BUILD (driver): full-box campaign v2 on the fixed trainer;
   census v2 (serve fraction, lnL tiers, binning floor with real
   statistics); slow sweeps via post_build_sweeps.sh.
