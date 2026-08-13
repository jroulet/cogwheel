---
section: Backlog
---

- **THE FOLD-PPGO GATE IS INVERTED IN `xi`, AND THE CORRECTION SHOULD BE
  DELETED FROM THE RUNG — BLOCKED ON A RAW-PPGO CERTIFICATE** `[→ spec]` —
  measured 2026-08-13 (325 config-w points, gamma in {0.3, 0.5, 0.7}, oracle
  = direct `f_schwinger` validated at 7.7e-17 against `exact_total`).
  Supersedes [[lensing_fold_ppgo_rung_serves_wrong]]'s open question; see
  [[FINDINGS F072]] for the refuted derivation and the cusp-tie bug.

  ## What is settled

  The rung already serves RAW ppGO (71a5051), so the wrong ANSWER is gone.
  What remains is the gate:

      shipped:  rho <= 1  AND  xi_min >= 4  AND  est <= CERTIFICATION_BAR
      the win:  xi_min <= 1.0   (crossover xi = 0.89-1.09, 20% spread,
                                 stable across 2x in w and 2.3x in gamma)

  The gate demands the EXACT INVERSE of the region where a fold correction
  helps. The third leg (`est <= 1e-4`, i.e. `w*dtau >= 13344*c_A`) pushes the
  same wrong way.

  **A fold-corrected rung is not worth having.** Best fold error anywhere on
  325 points is 2.15e-3 — 21x over `CERTIFICATION_BAR`; median on fold arcs
  4.0e-2, 400x over. There is NO interior configuration where the fold
  correction certifies. The win region is under 1% of the interior at the
  `w_lo` this rung actually operates at (35% at w=60, 0.33% at w=5e4), and
  inside it neither arm certifies. Where the rung DOES serve (`xi >= 4`) raw
  ppGO is already 2.0e-5..2.5e-4, improving as `w^-2.18`.

  **`rho` cannot be a gate coordinate here.** `caustic_rho` normalises by the
  MAX caustic reach (attained on the cusp axis), so along `theta = pi/4` at
  `gamma = 0.5` the caustic sits at `caustic_rho = 0.354`, not 1. `rho <= 1`
  is close to vacuous and NO rho threshold locates the caustic. An earlier
  note in this backlog said the correction "wins for rho >= 0.93" — that was
  the DIRECTIONAL gauge, not the gate's gauge. Corrected here.

  ## Why the obvious change was NOT made

  The recommendation is to drop the `_uniform_error_estimate` leg. That is a
  THREE-ORDER-OF-MAGNITUDE widening: the leg requires
  `w*dtau >= 13344*c_A`, while `xi >= 4` alone is `w*dtau >= 10.7`. Raw ppGO
  at `xi >= 4` measured 2.0e-5..2.5e-4 — which STRADDLES the 1e-4 bar at the
  low end. Dropping the leg would therefore serve some configs over the bar
  with no certificate at all.

  BLOCKER: the rung needs a bound derived from RAW PPGO's own asymptotics
  (measured `w^-2.18` here), not one inherited from the fold arm. Until that
  exists, the mis-shaped gate is at least CONSERVATIVE — it serves rarely,
  and what it serves is raw ppGO, which is accurate there.

  Do that derivation first, then drop the leg. Do not drop it alone.

  ## Acceptance

  A certificate for raw ppGO on this rung's domain, demonstrated where the
  exact engine can check it (`w <= 60`), plus the extrapolation stated
  explicitly — the rung's real `w_lo` is 1e3-1e5 and NO oracle in this repo
  reaches there, so the extrapolation is the residual risk and must be named,
  not hidden.
