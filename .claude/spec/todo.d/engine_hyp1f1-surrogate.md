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

SECOND LEVER — exploit the factorization h_L = F * h_UL (owner,
2026-07-17): F(f) is far smoother than h_UL(f), so the kernels deserve
their OWN coarse global node grid (splined/interpolated to bin centers),
decoupled from the waveform's 253 bins — currently the kernels inherit
the bin grid (506 engine points/eval), paying h_UL's resolution for a
quantity that needs a fraction of it. The delay phases stay analytic
(already exact). Multiplicative with the surrogate: fewer evaluations x
cheaper evaluations. Measured split (2026-07-17, crown 4-image config):
engine 19.36 s (99.3%), contraction 0.142 s, ratio path 1 ms — after
both levers the 142 ms contraction becomes the next target.

Acceptance: lnlike within the existing accuracy gates (crown RB-vs-brute
at original tolerances; closed-form macro gate; certification battery)
with the surrogate in the hot path, and measured eval time in the few-ms
range. Surrogate-vs-DD-oracle error must be gated explicitly (tolerance
with provenance, F002-safe: oracle stays the DD ladder / mpmath).
