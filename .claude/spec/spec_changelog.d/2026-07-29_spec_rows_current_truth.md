---
date: 2026-07-29
bump: patch
---

### Microlensing SPEC rows rewritten to current truth only

The engine and sampling-layer rows had accreted build-by-build narrative:
retired mechanisms described as live, corrections stacked on top of the claims
they corrected, and a stale limitation contradicting the rest of the row.
Rewritten to state what is true now. No described behavior changed.

Corrected, not merely trimmed:

- The engine row opened with "positive-parity macro images only; macro saddles
  out of scope" while the same row went on to describe the saddle wave branch,
  saddle channels, and saddle integration. Both parities serve.
- The batched fast path was described as the per-order weight-vector
  contraction over an 85x85 bilinear form -- the retired operator series.
- `CancellationError`, `legacy_operator_oracle`, `_fused_contraction`,
  `half_sum` and `_SERIES_TOLERANCE` were named as live mechanisms; none
  exists.
- Two live modules were missing entirely: `_born.py` (the far-annulus Born
  carrier, shipped but not wired) and `_pearcey_table.py`.
- F019's warning was restored in the text rather than by reference: two
  distinct ceilings both equal 60 (`W_CEILING_SCHWINGER` on frequency alone,
  `DD_PRODUCT_CEILING` on the product `w*sqrt(s)`).
- The `eta` leg is now stated as live on both parities, and
  `_certify_geometric_census` is named as NOT covering the near-caustic tail,
  which is the reason the leg exists.
