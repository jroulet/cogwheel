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

  Once the residual chart `F_exact - F_carrier` is driver-trained
  (TRAIN_TIER artifact — not yet built), re-derive the registration/accuracy
  gate in the residual currency and remove the fall-through at the fact-4
  slot in `likelihood.py::_surrogate_coefficients` (the comment there marks
  it), wiring the trained reconstruction through
  `channels.born_carrier_from_partition`. Until then the annulus stays
  exact-served — correct, just not zero-quadrature.

  Saddle branch: see [[lensing_saddle_born]].
