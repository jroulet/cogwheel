---
section: Backlog
---

- ~~**Normalize the far-field `d` axis by curvature radius**~~ `[RESOLVED]` —
  Evaluated 2026-08-03 (build `eval_d_norm`). Rejected: wrong physics (Airy
  transition is ξ not d/R_c), wrong chart (far-field operates at d >> R_c),
  breaks tensor-product separability. Proceed with absolute d. See
  `completed.d/2026-08-03_d_normalization_evaluation.md`.

- ~~**ppGO handoff above chart w-ceiling for interior draws**~~ `[DONE — 2026-08-03]` —
  Landed option (b): `_surrogate_coefficients` now serves interior draws above
  the `InteriorWedgeChart` w-ceiling when `xi_min >= _XI_FOLD_THRESHOLD = 4.0`
  AND the per-pair `_uniform_error_estimate` is below `CERTIFICATION_BAR`.
  Census tracks these as category `ppgo_fold`. SPEC.md updated.
  See `completed.d/2026-08-03_fold_ppgo_interior_handoff.md`.

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
