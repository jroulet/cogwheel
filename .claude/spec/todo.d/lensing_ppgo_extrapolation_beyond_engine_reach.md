---
section: Backlog
depends_on: [lensing_exterior_followup_four_items]
---

- **SERVE ppGO WHERE THE EXACT ENGINE CANNOT REACH, INSTEAD OF REFUSING**
  `[→ spec]` — owner-directed 2026-08-06. Distinct from the ppGO ROUTING item
  in [[lensing_exterior_followup_four_items]] (which is about choosing ppGO
  where it is already available); this is about EXTENDING coverage into
  `(y, w)` regions the engine's own ceilings forbid.

  THE CONSTRAINT: charts can only be trained where the engine can be called.
  `f_schwinger` refuses above `W_CEILING_SCHWINGER_QD = 150`, the dd path caps
  at `w <= 60`, and the 1F1 kernel caps the PRODUCT `w*sqrt(s) <= 60`. So a
  chart's `w_max` is pinned by `_DD_PRODUCT_MARGIN / (r_max * reach_max)`
  regardless of whether the PHYSICS is hard there. It is not: at large `w` and
  well-resolved images the field is MORE classical, not less. We refuse
  exactly where the asymptotics are best.

  THE IDEA: in the region where images are well-resolved, ppGO (post-post
  geometric optics) is a good extrapolation, and its accuracy IMPROVES with
  `w`. Serve it there under a named rung rather than falling through to a
  refusal. The fold-ppGO interior handoff already does exactly this for one
  case (`xi_min >= _XI_FOLD_THRESHOLD = 4.0` plus a per-pair uniform error
  estimate below `CERTIFICATION_BAR`) — generalise that pattern.

  OWNER'S SUGGESTION, worth testing: where a direct high-`w` fit is
  impossible, use the INTERIOR FITS PLUS A SCALING. The `w`-dependence in the
  resolved regime is largely carried by known analytic factors
  (`exp(i w tau_a)` carriers, `w^{1/6}`/`w^{-1/6}` Airy weights, the `w^{1/2}`
  / `w^{3/4}` Pearcey control arguments), so a chart trained in the reachable
  band may extrapolate in `w` once those are divided out — i.e. spline the
  DEMODULATED, scaling-stripped residual, which is `w`-flat by construction,
  rather than the envelope itself.

  ACCEPTANCE: name the region (in `rho`, `theta`, `w`) where ppGO is served;
  certify against the exact engine INSIDE the reachable band with the same
  eps currency used elsewhere (p50/p90/max plus worst-sample locus); show the
  error DECREASES with `w` across that band, which is the only honest basis
  for trusting it beyond the ceiling; and make the extrapolated region a
  NAMED rung in the serving ladder, never a silent extension of a chart's box.

  DO NOT let this become an unfenced arm. `_airy_fold`'s fence is permanent
  precisely because a self-certificate that cannot see distance from the
  caustic read 1.2e-2 where the true error was O(1) (F028, F032, F033). Any
  ppGO extension needs a gate keyed on the thing that actually degrades it.
