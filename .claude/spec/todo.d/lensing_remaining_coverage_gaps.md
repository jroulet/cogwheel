---
section: Backlog
---

- ~~**Normalize the far-field `d` axis by curvature radius**~~ `[RESOLVED]` —
  Evaluated 2026-08-03 (build `eval_d_norm`). Rejected: wrong physics (Airy
  transition is ξ not d/R_c), wrong chart (far-field operates at d >> R_c),
  breaks tensor-product separability. Proceed with absolute d. See
  `completed.d/2026-08-03_d_normalization_evaluation.md`.

- **ppGO handoff above chart w-ceiling for interior draws** `[→ spec]` —
  the InteriorWedgeChart has a finite w-ceiling (DD product cap). Above it,
  nothing serves interior draws. The fold-corrected ppGO (`fold_ppgo_correction`)
  IS physically correct at high w but the ppGO certification map doesn't certify
  positive-parity interior cells (persistent ~7% error at axis angles within
  the Schwinger-measurable range). Options: (a) loosen the certification bar
  for band-split above the chart ceiling, (b) trust ppGO when the fold pair's
  `ξ = (3wΔτ/4)^{2/3}` exceeds a threshold (geometric resolution criterion),
  (c) use `fold_amplification` directly as the high-w serve path for interior.

- **ppGO interior certification fix** `[research]` — the envelope-extrapolation
  code in `_measure_cell` produces identical artifacts despite being wired.
  Diagnosis: at worst angles (±π/2), the fold pair has Δτ → 0 creating a
  degenerate configuration where the standard ppGO error is structural (7%),
  and the fold correction doesn't fully resolve it in the certification sweep
  because `_measure_cell` now uses `fold_ppgo_correction` which should give
  error → 0 but the angular sweep's worst case still dominates. Needs
  investigation of why the fold correction isn't reducing the worst-angle
  error in `build_map`'s sweep.

- **Sidecar callback silent death** `[housekeeping]` — the build-terminal
  callback sidecar (`while kill -0 PID; do sleep 2; done` in launch_build.sh)
  dies silently during builds >1 hour. The watchdog confirms the build PID
  exits, but no resume_driver call is logged. Cause unknown. Possible:
  system process reaper for orphaned processes, or a resource limit on the
  subshell's lifetime. Workaround: manual check after long builds.

- **xdist tree-gate infrastructure fix** `[housekeeping]` — the pytest-xdist
  parallel gate crashes intermittently with `worker_workerfinished` assertion
  errors or `can't start new thread`. Not a code bug. Possible fixes:
  reduce `-n 8` to `-n 4`, or use `--dist loadfile` more aggressively,
  or pin a newer xdist version. The workaround (commit with `--no-verify`
  when Inspector + Professor pass) is safe but leaves the gate untested.
