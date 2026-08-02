# Coder Short-Term Observations

- wp1: Added `except CarrierDiscontinuityError as exc:` clause to `_eps_for`
  inner function of `_reprovision_w_nodes` in surrogate_training.py (after
  line 3110). The clause catches degenerate-tile topology refusals from
  `_build_farfield_chart` and records status='carrier_discontinuity' with
  detail in the trace, returning None to trigger the safe fallback path.
  `_ENGINE_REFUSALS` tuple left unchanged.
