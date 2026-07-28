# Architect Short-Term Observations

## Born carrier + band-split (brief_born_carrier_bandsplit, 2026-07-28)
- Wire Born rung, far annulus 3.0<|y|<=4.2426, positive parity, no quadrature.
- `_born.py` currently DORMANT; likelihood.py born slot (~L1650) returns None.
- Fixes: `_born_factors` add a0 (5-tuple), b1 sign fix; correction += a0/q2r;
  born_gate guard A rescale by b1**2 + re-key to w*r0_sq; docstring backwards.
- Band split at w_split keyed on w*r0_sq<~8 (named const, settable). LOW band =
  born carrier alone; HIGH band w>=w_split = EXISTING far-field ppGO+ghost
  machinery (geometric_amplification + farfield_ghost_term / kernel-sum-minus-ghost).
- Census: add 'born' category to classify_fallthrough + _FALLTHROUGH_CATEGORIES.
- OUT: macro-saddle (gamma>1), low-w analytic rung, cusp balls, census RUN,
  TRAIN_TIER shipped artifact (driver-owned). In-build tests FAST only.
- Real gate = residual node counts within ~2x of F023 table on small synth config.
