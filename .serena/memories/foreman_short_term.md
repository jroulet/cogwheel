# Foreman-Lite Short-Term Observations

## 2026-08-20 (INS-1-002, low_w_diffractive_chart docstring)

- DOC-FIX CANONICAL-SOURCE RULE: when a chart's module/evaluate docstrings are
  stale but the likelihood serve docstring (and the training script) describe
  the representation correctly, take the formula VERBATIM from the training
  script (`r_pure = f_pure / (sqrt(mu_pure) * prefactor_c(w))`) and the serve
  (`F_serve = mass_sheet_phase * prefactor_c(w) * sqrt_mu_full * r_pure`) —
  never re-derive a paraphrase. Here the stale "anchor" framing
  (`F_serve = r * anchor`, anchor = single factor sqrt(mu_macro)*exp(phase))
  was subtly wrong because the real normalization is a PRODUCT of two analytic
  factors (sqrt(mu_pure) AND prefactor_c(w) = C(w), |C| ~1.4 at w=0.5); the
  fix names both. Same-class trap as the 2026-08-08 "paraphrase grep miss"
  entry: grep the EXACT string from the finding, not a paraphrase.
- Verified SPEC.md has NO stale claim about this chart's residual (the
  SPEC's `F(w->0) = sqrt(mu_macro)*exp(-i*pi*n/2)` refers to the analytic
  Rung P anchor in `_diffractive.py`, a different object — correct as-is);
  no docs/source/ edit -> no Sphinx rebuild needed.
- Verify recipe for doc-only fixes: ast.parse (pyflakes absent in
  cogwheel-newlal) + live import, then a read-back of both edited regions.
  No test run needed — zero code behavior changed.
