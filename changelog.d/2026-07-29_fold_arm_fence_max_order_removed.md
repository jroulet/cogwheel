---
date: 2026-07-29
---

### `max_order` removed from the public microlensing API; fold arm gains a caustic-distance fence; macro saddle gains its own eta floor

`max_order` is removed entirely — from `operator.F_op`, `F_op_grid`,
`ChangRefsdalChannels.__init__`, and
`LensAmplificationSurrogate.from_engine`/`from_lobe_engine`. It was a
parameter of the legacy dd/1F1 operator-series contraction retired in the
prior release; nothing reads it anymore. Callers passing `max_order=...`
to any of these must drop the argument. `operator.MAX_ORDER` and the
now-orphaned `_MIN_ORDER`, `_CONSECUTIVE_SMALL`, `_SERIES_TOLERANCE`
module globals are deleted too.

The uniform fold (two-image) Airy arm now refuses to serve beyond
`eta = 0.3` from the caustic (`operator.ETA_MIN_GEOMETRIC`'s complement),
where it was measured 60%-267% wrong (F028) and independently confirmed
63%-64% wrong against GLoW (F032). This is a coverage-for-correctness
trade: nodes failing the fence now fall to the exact Schwinger evaluator or
a named refusal instead of a wrong served value; measured cost is up to
~10% of sampled draws.

The macro-saddle branch of the geometric-vs-wave gate (`select_branch`) now
also measures its distance to the caustic and refuses the geometric branch
below `eta = 0.3`, matching the positive-parity gate. It previously passed
`eta = inf` (the leg off) because no saddle-specific measurement existed;
that default was found to serve up to 484x error over 15% of resolved
draws (F034) and has been corrected.
