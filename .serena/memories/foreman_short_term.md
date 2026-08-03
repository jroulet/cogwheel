Last build: measure_dropped_slivers script (region 10).

- Created `scripts/measure_dropped_slivers.py`: calls `stable_gamma_bands` over the
  full prior sub-range for both parities (positive: 0.0–0.999, saddle: 1.001–1.6)
  with min_width=0.02, sums dropped widths, reports CLOSED/OPEN verdict vs 1e-3.
- Pattern: lightweight loop, direct function call, per-parity print then global summary.
  Stability advisory: reruns at n_samples=400 and warns if dropped list differs.
- Verification: AST OK, import OK from cogwheel-newlal env.
- No production code modified; no callers affected.
- SPEC mentions dropped_gamma_slivers correctly in training provenance context —
  no staleness introduced.
- Observation: `stable_gamma_bands` returns `(stable, dropped)` where each dropped
  entry is a `(lo, hi)` float tuple — simple to sum with a generator expression.
