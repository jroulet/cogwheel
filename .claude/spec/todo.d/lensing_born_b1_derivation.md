---
section: Backlog
---
- **Wire the Born carrier + band-split residual charts into the serve path**
  `[→ spec]` — derivation AND implementation landed 2026-07-28 (commit
  `31ee133`, FINDINGS F023-F026): `_born_factors` returns the derived
  closed-form `b1`/`a0`; `born_lead_carrier` is the lead-only serve object;
  `channels.born_carrier_from_partition` assembles the band split at
  `w * Delta_tau = RHO_END` (read from the partition, never recomputed);
  guard A is re-keyed to that same split and the module gains the exact
  `gamma < 3/4` exterior fence; the `'born'` census category is landed in
  `surrogate_census.classify_fallthrough` (annulus draws no longer
  mis-attributed to `out-of-box`). What remains is the LAST step only:

  **C8 blocker discharged.** Coordinates restated in `rho`; annulus region
  artifact retired. **C11 landed** (2026-08-01): `BornResidualChart` frozen
  dataclass (`cogwheel/lensing/born_residual_chart.py`) defined; fact-4 slot
  in `likelihood._surrogate_coefficients` wired to conditionally serve when
  a chart is attached. When chart is `None` (default), annulus still falls
  through to exact engine — correct.

  **What remains — TRAIN_TIER only:** once the residual chart
  `F_exact - F_carrier` is driver-trained, attach the `BornResidualChart`
  instance to the likelihood object to enable zero-quadrature serving. The
  fall-through is the correct behavior until then.

  Saddle branch: see [[lensing_saddle_born]].
