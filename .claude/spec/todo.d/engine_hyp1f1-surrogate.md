---
section: engine
---
# Fast hypergeometric evaluation — tabulation/surrogate [→ spec]

Owner directive (2026-07-17): the lensed-RB performance target is FEW
MILLISECONDS per lnlike — the standard cogwheel relative-binning regime
for unlensed frequency-domain models. Measured now: ~20 s/eval (nsub=2,
506 engine points), entirely dominated by naive per-point 1F1 evaluation
(the DD shared-numerator series ladder in _hyp1f1.py).

The lever is NOT (only) fewer evaluation nodes — it is the cost of EACH
evaluation: naive 1F1 calls are very inefficient. Build a tabulation or
surrogate for the channel kernels / 1F1 over the certified domain
(w x effective-shear x parity band), e.g. precomputed tables with
certified interpolation error, or a Chebyshev/spline surrogate, with the
DD ladder retained as the accuracy oracle and for out-of-table refusal.

Acceptance: lnlike within the existing accuracy gates (crown RB-vs-brute
at original tolerances; closed-form macro gate; certification battery)
with the surrogate in the hot path, and measured eval time in the few-ms
range. Surrogate-vs-DD-oracle error must be gated explicitly (tolerance
with provenance, F002-safe: oracle stays the DD ladder / mpmath).
