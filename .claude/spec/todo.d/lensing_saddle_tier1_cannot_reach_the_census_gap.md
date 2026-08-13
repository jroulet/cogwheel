---
section: Backlog
---

- **THE TIER-1 SADDLE RUNG AND THE MEASURED SADDLE GAP ARE DISJOINT BY
  CONSTRUCTION — COVERAGE DID NOT MOVE, AND COULD NOT HAVE** `[→ spec]` —
  measured 2026-08-13 by the driver, `census_dry_run.py --n-samples 10000
  --seed 42`, by spying on `_classify_saddle`.

  Structural coverage after the tier-1 build: **87.61%** — byte-identical to
  before it. Not a regression and not a wiring bug alone: the two populations
  cannot intersect.

  ## The measurement

  All 1742 draws that reach `_classify_saddle`:

      exact_engine (the gap) 1236   rho p10 0.086  p50 0.445  p90 0.836  max 0.999
      lobe_exterior           494   rho p10 0.094  p50 0.556  p90 0.919  max 1.000
      lobe_interior            12   rho p10 0.151  p50 0.402  p90 0.735  max 0.740

      rho >= _SADDLE_FARFIELD_RHO_FLOOR (2.0):  0.00%  in ALL THREE

  ## Why it is structural, not incidental

  `classify_draw` routes `if rho > 1.0: return 'born'` (L275) BEFORE
  `if gamma >= 1.0: return _classify_saddle(...)` (L283). So the saddle path
  only ever sees `rho <= 1` by construction, and the tier-1 gate demands
  `rho >= 2`. No tiling, no threshold tuning, and no amount of census
  re-wiring changes this: the ceiling on tier-1's contribution to the
  measured saddle gap is exactly ZERO.

  This also corrects the reading that motivated the build. The gap's large
  `rho_lobe` (p50 ~4.9) is the LOBE-LOCAL radius, divided by the small
  `r_deltoid`. The caustic-relative `rho` — the gauge the serve floor uses —
  is p50 0.445. These sources are NEAR the caustic in the gauge that governs
  tier-1 accuracy, which is precisely why a zero-envelope serve is wrong for
  them and the floor refuses them. Two radii named `rho` in one system, one
  small where the other is large; see [[lensing_saddle_forensics]].

  ## Consequence for tier 2 — this SHARPENS the target

  The entire 1236-draw gap lives at `rho <= 1.0`, with p50 0.445. That is
  exactly the deferred tier-2 window (the near-caustic resolvable saddle),
  and tier 2 is now the ONLY rung that can close any of it. A tier-2 chart
  must cover `rho` in roughly (0.09, 1.0] to reach the p10-p100 span, not
  the far exterior.

  ## WP-2's census wiring is also in the wrong module (separate defect)

  WP-2 wired the rung into `surrogate_census.characterize_sample`, but
  `scripts/census_dry_run.py` NEVER calls that function — it classifies
  saddle draws through its own `_classify_saddle` (L136, called at L284).
  Measured: 0 gate calls from a full census run. Even correctly placed it
  would attribute 0 (see above), so this is a latent-correctness issue rather
  than a lost-coverage one, but the build's WP-2 acceptance ("the census
  attributes the rung") is UNMET and should not be recorded as delivered.

  Note `_classify_saddle(gamma, y_abs, theta)` takes no `w` and no geometry
  partition, so the resolvability term cannot be evaluated there as written;
  wiring it properly needs the classifier to carry the band floor.

  ## What the rung IS worth

  Production dispatch is surrogate -> ppGO(`w_max > 150`) -> tier-1 -> exact
  seed engine, with no Born intercept, and the surrogate is OFF by default.
  So tier-1 does fire for a far-from-caustic (`rho >= 2`), resolvable,
  `gamma > 1`, `w_max <= 150` candidate whenever the surrogate is absent or
  declines, replacing an exact engine call with a zero-envelope analytic
  serve at p90 ~5e-5. That is real, it is tested, and it is simply not the
  saddle gap.

  ## Acceptance

  Do NOT claim a coverage delta for tier-1. Re-run the six-way breakdown and
  report 87.61% unchanged with this disjointness as the reason. The next
  build that claims saddle-gap coverage must first show its rung's admitted
  domain OVERLAPS `rho <= 1`.
