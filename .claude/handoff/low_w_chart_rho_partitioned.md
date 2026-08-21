# Build: rho-partitioned low-w diffractive chart — geometric/macro carriers for the wall band

## Mission

The cusp-fallback smoke bake exposed a fundamental design error (Professor
ruling, 2026-08-21, recorded in `.claude/handoff/low_w_chart_rho_hierarchy.md`):
the far-exterior wall band (gamma' > 0.5, 57.5% of engine residual) is NOT a
fold or cusp. At gamma'=0.8 the caustic reaches |y| ~ 3.58; the wall-band
cells (rho up to 7.2, e.g. rho=2.0) are far exterior with two well-separated
images — no fold, no cusp. The `b3->0` signal meant the fold form's
MERGING-PAIR assumption broke down, not "there is a cusp here." Using
fold/cusp normal forms as the chart carrier there produced a 5800x residual
pole (de-rate 0.0002, every cell declined).

The chart must use a RHO-PARTITIONED CARRIER HIERARCHY, following the Born/
ppGO exterior ladder's representation (F023-F025): the wall band is served by
the geometric/macro ladder, the caustic neighborhood by the fold/cusp forms.

## Correct hierarchy (binding, Professor ruling)

1. CAUSTIC NEIGHBORHOOD (rho ~ 1): Airy fold (q=p) Wronskian carrier, with
   the Pearcey fallback ONLY in the genuine cusp window (b3->0 AND rho ~ 1,
   where the 3 images are cluster-resolved and cluster_sum is O(1)).
2. FAR EXTERIOR, RESOLVED (rho >= rho_split, w*Delta_tau >= few): the
   TWO-IMAGE GEOMETRIC-OPTICS SUM carrier — exact where images are well-
   separated; the residual against it is the smooth O(1) diffractive
   correction.
3. FAR EXTERIOR, UNRESOLVED (rho >= rho_split, low w): the MACRO LEAD
   CARRIER sqrt(mu_macro) = 1/sqrt((1-kappa)^2 - gamma^2) — ALREADY the
   residual's normalization (f_pure*sqrt(1-gamma'^2) -> 1 at low w). The
   diffractive correction is the smooth residual. This is exactly the
   `born_lead_carrier` pattern (`_born.py`: sqrt(mu_macro)*exp(i w phi_geo)).

The rho-partitioning is the DRY single source: a cell's carrier is chosen by
its rho band (with the resolved/unresolved split by w*Delta_tau at serve),
shared verbatim between the trainer and the serve.

## Two requirements (without which the pole returns)

- ABSOLUTE carrier-adequacy guard (NOT the self-referential min/max spread
  guard): bound |F_ref| / |f_pure*sqrt(1-gamma'^2)| (or directly |r|) on the
  w-grid; a carrier whose residual is 3-4 orders off-normalization is
  declined, never de-rated into submission.
- MAGNITUDE: normalize the Airy F_ref to the macro limit at low w (its
  w^{-1/6} divergence forces r->0, not r->1), or fold the macro lead into
  the reference, so the fold-side residual is genuinely O(1) at the band
  bottom.

## Scope

IN:
- `cogwheel/lensing/low_w_diffractive_chart.py`: `fold_cusp_reference` gains
  the rho-partitioned hierarchy — caustic-neighborhood cells use Airy (+
  restricted Pearcey), far-exterior cells use the geometric two-image sum
  (resolved) or the macro lead `sqrt(mu_macro)` (unresolved). One reference
  builder, cell-partitioned by rho, DRY.
- `scripts/train_low_w_diffractive_chart.py`: the residual target and the
  declined/unbuildable classification use the partitioned reference; the
  absolute carrier-adequacy guard replaces the spread guard; provenance
  reports per-carrier populations (fold/cusp/geometric/macro).
- `cogwheel/lensing/likelihood.py`: `_low_w_diffractive_chart_serve`
  re-modulates with the SAME partitioned carrier the trainer used (the
  resolved/unresolved split by w*Delta_tau at serve).
- Tests: per-carrier served-accuracy pins (fold cell, cusp cell, geometric
  far-exterior cell, macro far-exterior cell each |F_serve - F_engine|/
  |F_engine| <= 1e-4 at served w); the absolute guard's self-falsification
  (a carrier 3-4 orders off-normalization is declined); the rho-partition
  continuity (no step at the rho_split boundary).

OUT (do not touch):
- The order-16 series, `w_low_fit`, the fence, Rung S.
- `_airy_fold` / `_pearcey_cusp` internals (import, don't modify).
- The existing serving ladder's geometric/ppGO/Born arms (import their
  carriers; don't re-implement).

## Acceptance

- Smoke bake: de-rate far above 0.0002 (report the number), served error
  approaching 1e-4 on the smoke grid + off-grid midpoints for ALL carriers,
  no residual pole (the absolute guard declines ill-conditioned cells, never
  de-rates them).
- The wall band (far exterior) is served by the geometric/macro carriers,
  NOT the fold/cusp forms; the caustic neighborhood by Airy (+ restricted
  Pearcey).
- The rho-partitioning and the resolved/unresolved split are DRY single
  sources shared by trainer and serve.
- Full bake + shipped npz = DRIVER step.

## Constraints

- Branch `claude-dev`. Slow tiers stay gated.
- Spec/TODO workflow: `[→ spec]` + completion record; `lensing_low_w_near_
  fold_serve` binding.
- The full bake + held-out validation is a DRIVER step.
