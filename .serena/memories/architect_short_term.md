# Architect Short-Term Observations

(last entry 2026-08-05)

- Build brief_dropped_slivers.md: pure measurement script `scripts/measure_dropped_slivers.py`. Calls `stable_gamma_bands` over full prior sub-ranges per parity with production min_width=0.02. Professor predicts dropped=[] (zero slivers — no real metamorphosis in the Chang-Refsdal model across the prior). Simplifier says single WP, trim n_samples=400 confirmation (kept as lightweight advisory per Professor). Region 10 expected to close trivially.

- Build brief_1b_training_path_consumers.md describes work already completed in commit 00bf8ae (2026-07-29, "F041 arc-guard fix + salvaged 1b estimator retirement"). All six numerical estimators in surrogate_training.py were retired against the analytic geometry cascade. The brief file was created on 2026-07-31 (commit b712709) as a stale handoff — it documents the spec for already-completed work, not pending work.

- Build brief_1c_cusp_vertex_y3.md describes work already completed in commit b9c3ed6 (2026-07-30, "lensing: 1c -- analytic cusp vertex on the serving path + y''' (third order)"). All acceptance criteria met: _cusp_vertex uses brentq on analytic g=y'.y'' (O(1) calls), caustic_third_derivative exists as separate function, tests pass (25+63 green). The brief was created AFTER the work completed. TODO.md step 1c lacks a DONE marker — Librarian scope only.
