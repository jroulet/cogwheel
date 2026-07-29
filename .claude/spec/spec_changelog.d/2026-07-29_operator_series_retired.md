---
date: 2026-07-29
bump: minor
---

### Legacy operator-series contraction fully retired; `CancellationError` deleted; `select_branch` gains an eta (distance-to-caustic) leg

SPEC.md described the shear-free `gamma' == 0` point-lens exit as still running
the legacy dd/1F1 operator-series contraction, with `CancellationError` as a
live refusal in the wave-branch vocabulary. Both are gone: the shear-free exit
now serves a byte-identical CLOSED FORM (the shear operator is the identity at
`gamma' = 0`, so the series collapsed to its zeroth term, which is exactly the
point-mass kernel); the legacy contraction survives only for test-only oracle
duty (`operator.legacy_operator_oracle`). `CancellationError` was raised only
inside the deleted contraction, so it is deleted too — removed from every
named-refusal vocabulary list, docstring, and except-tuple across the engine
and sampling layers.

Also corrected: `operator.select_branch` (the authoritative geometric-vs-wave
gate documented in the prior authoritative-gate entry) gains a third leg —
`eta >= ETA_MIN_GEOMETRIC = 0.3`, distance to the caustic — alongside
resolution and cancellation. FINDINGS F031 measured the two-leg gate admitting
p90 = 117% relative error at `eta < 0.1`; the third leg cuts worst-case p90 to
7.65e-5. This is a refusal-increasing change: nodes failing the eta leg now
fall to the uniform arms or the named refusal instead of a geometric serve.
The macro-saddle branch passes `eta = inf` (F031 is measured positive-parity
only, so its boundary is not inherited unmeasured).

`operator.MAX_ORDER` is unaffected by this entry — it was already vestigial
(a parameter default threaded through `F_op`/`F_op_grid`/`ChangRefsdalChannels`
but not consumed by any surviving series) and SPEC.md never named it directly.
