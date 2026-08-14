---
section: Backlog
---

- **`_CUSP_ARM_COVERAGE = 0.07` DERIVATION FALSIFIED BY F074 — RE-MEASURE
  BEFORE THE NEXT SURROGATE TRAINING RUN** `[→ spec]` — the constant is
  documented as the minimum image-theta offset at which the cusp arm
  serves, "measured by scripts/measure_cusp_arm_actual_boundary.py". With
  the corrected control map the arm serves AT the vertex (measured minimum
  offset 0.0 over 50 served sources; the admission no longer reads an
  angle). The constant still shrinks the tube cusp-exclusion window
  (surrogate.py:2891), so tube tiles may be sized on a stale boundary.
  Re-run the measurement script against the F074 arm, re-derive or retire
  the constant, and update the tube-window consumer. BLOCKS the surrogate
  training campaign config (the tube/cusp tiling reads it). F078 records
  the discovery context.
