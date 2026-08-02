# Architect Short-Term Observations

- Build: fix_carrier_discontinuity — CarrierDiscontinuityError unhandled in
  `_eps_for` inside `_reprovision_w_nodes` (surrogate_training.py:3107).
  Fix: add a separate except clause catching CarrierDiscontinuityError,
  returning None + trace entry with status='carrier_discontinuity'.
  Simplifier confirmed lean (1 WP, no module-level _ENGINE_REFUSALS change).
