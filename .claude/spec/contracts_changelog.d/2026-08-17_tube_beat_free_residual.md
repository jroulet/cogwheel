---
date: 2026-08-17
bump: minor
---
Updated the `lens_amplification_surrogate` contract's TubeChart description to
record the BEAT-FREE RESIDUAL representation and the new tube
envelope-definition tag (values-not-paths schema note; no producer path or
consumer wiring change).

- The TubeChart `real_coeffs`/`imag_coeffs` spline tensors now encode the
  residual `r(w) = E(w) / F_ref(w)`, not the raw near-caustic envelope `E(w)`.
  The near-caustic envelope carries two fold-pair carriers `exp(i*w*tau_pm)`
  that interfere as `cos(w*Delta_tau)`; no reparametrization of the
  interpolation axes removes that beat, so the fix is at the level of the
  stored REPRESENTATION.
- `F_ref(w)` is an analytic, non-vanishing Airy-uniform two-fold-carrier
  reference built via `airy_fold_value` with EQUAL amplitude arguments
  (`q = p`), giving `|F_ref|**2 ~ w**(1/3)*Ai**2 + w**(-1/3)*Ai'**2`
  (non-vanishing by the Airy Wronskian, so `r = E / F_ref` is always finite).
  The merging-pair delay-split `Delta_tau` and the shared t_min-relative frame
  are DRY-imported from `_airy_fold` (`_merging_fold_pair`), never re-derived.
- Each TubeChart record MUST carry the `envelope_definition` tag
  (str, default `'tube_beat_free_airy_v1'` = `TUBE_BEAT_FREE_AIRY`, the ONLY
  known tube envelope-definition tag). An absent or unknown tag hard-refuses at
  load via `_validate_tube_envelope_definition` (mirroring the far-field
  envelope-definition guard), so a stale raw-envelope tube artifact cannot
  silently mis-serve as a residual.
- `serve` reconstructs the physical envelope `E = r * F_ref` (F_ref recomputed
  at the query w from the raw source eigen-coordinates, D2-invariant). The
  engine-free tube census is therefore apples-to-apples and needs NO arithmetic
  change: `tiling_census._count_tube` counts fold ARCS (an internal
  representation change is invisible to it) and `surrogate_census`'s held-out
  eps compares the served physical `E` against the engine reference
  `partition.envelope` normalized by `max|E|`.
- The `theta_to_s` arc-length axis-map is unchanged; only the coefficient
  arrays' MEANING (residual, not envelope) is now tagged by
  `envelope_definition`.
