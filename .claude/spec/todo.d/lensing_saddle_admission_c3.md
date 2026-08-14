---
section: Backlog
---

- **NEXT-SESSION ORDER 2/7 — SADDLE EXTERIOR ADMISSION VIA THE c3+GHOST
  CERTIFICATE (subsumes the failed eta-floor build)** `[→ spec]` — build
  saddle_admission_predicates DIED at `error_max_turns` (2026-08-14 03:19,
  fix agent consumed its budget iterating the currency-corrected floor
  scan); its uncommitted gate/census/test changes were REVERTED to the
  green anchor. What survives: the currency-corrected
  `scripts/measure_saddle_eta_floor.py` (in HEAD), the fix agent's last
  script edits + plan + both escalations + full build log in
  `.claude/handoff/saddle_eta_build_salvage/`, and the coder checkpoint
  `refs/sdk/coder_checkpoint` (d5672fa6cd98) for cherry-picking.

  DO NOT re-run the eta-floor approach. Launch from
  `.claude/handoff/symmetry_tie_c3_admission.md`, which now carries BOTH
  objectives: (a) certificate admission (c3 + ghost,
  `geometry.ppgo_error_estimate` + `ghost_kernel` term) for ALL 2-image
  saddle exteriors — transverse cone, connecting region, symmetry axes —
  replacing the scalar-rho floor AND the eta proxy in one step; (b) the
  tie-discipline separation discriminator. Rationale on record
  (closed-form-before-estimator): eta was a scalar proxy for the two
  closed-form remainder pieces; three escalations chased the proxy's
  calibration currency, and the certificate computes the actual object
  per-draw in ~6 ms. The scan data doubles as the exterior-certificate
  calibration set — reuse, never re-scan.

  Build-ops lesson for the plan: heavy in-build measurement killed the
  last attempt at max_turns. Either raise the fix-agent turn budget for
  measurement WPs, or pre-compute the calibration DRIVER-SIDE and hand it
  in as a measured fact (preferred — briefs discipline already says
  measured facts the agents cannot cheaply obtain belong in the brief).
