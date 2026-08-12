---
date: 2026-08-12
bump: patch
---
### Pearcey arm: interior degenerate cluster bypass + on-axis fold detection

Extended the INTERIOR CUSP SERVING description in the Microlensing engine row:
the calibration bypass now covers two cases — (1) the generic interior case
(3 real stationary points, `rho < 1`) and (2) the interior degenerate cluster
(`rho < 1`, `len(images) > 2`, `len(stationary_values) == 1` due to first-order
control degeneracy on the cusp symmetry axis when the fold arm declines).
Added description of `_merging_fold_pair` detecting the degenerate cluster via
`_CUSP_TIE_EPS = 1e-12` (two saddles at tied delay returns None, routing to
cusp arm as last rung). Exterior sources (`rho > 1`) still validate
delay-to-image alignment.
