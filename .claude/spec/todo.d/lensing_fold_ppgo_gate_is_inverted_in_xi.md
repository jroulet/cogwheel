---
section: Backlog
---

- **THE FOLD-PPGO GATE IS INVERTED IN `xi`; THE RAW-PPGO CERTIFICATE NOW
  EXISTS, BUT IT MUST CARRY A GHOST TERM** `[→ spec]` — gate analysis measured
  2026-08-13 (325 config-w points); certificate derived and measured
  2026-08-13 (434 configs x 16 w-points = 6944 points, oracle = direct
  `_schwinger.f_schwinger` validated at 1.4e-16 against `exact_total`).
  See [[FINDINGS F072]] for the refuted eta-derivation and the cusp-tie bug,
  [[FINDINGS F069]] for why `F_op` is not an oracle above w = 60.

  ## What is settled

  The rung already serves RAW ppGO (71a5051), so the wrong ANSWER is gone.
  What remains is the gate:

      shipped:  rho <= 1  AND  xi_min >= 4  AND  est <= CERTIFICATION_BAR
      the win:  xi_min <= 1.0   (crossover xi = 0.89-1.09, 20% spread,
                                 stable across 2x in w and 2.3x in gamma)

  The gate demands the EXACT INVERSE of the region where a fold correction
  helps. **A fold-corrected rung is not worth having**: best fold error
  anywhere on 325 points is 2.15e-3 — 21x over `CERTIFICATION_BAR`; median on
  fold arcs 4.0e-2. There is NO interior configuration where the fold
  correction certifies.

  **`rho` cannot be a gate coordinate here.** `caustic_rho` normalises by the
  MAX caustic reach (attained on the cusp axis), so along `theta = pi/4` at
  `gamma = 0.5` the caustic sits at `caustic_rho = 0.354`, not 1. `rho <= 1`
  is close to vacuous and NO rho threshold locates the caustic.

  ## CORRECTION: the decay exponent is -3, not -2.18

  An earlier note in this backlog recorded raw ppGO improving as `w^-2.18`.
  **That number was contaminated and is withdrawn.** Two causes, both
  measured:

  1. It was fitted against `F_op`, which above w = 60 returns the uniform arm
     rather than an independent evaluation (F069). At gamma=0.5, rho=0.1 the
     `F_op` defect is ~7e-3 and does NOT decay, while the quantity being
     measured is ~1e-5 — a contaminant 1000x larger than the signal. Fitted
     over [8,150]: **-2.788 against `f_schwinger`, -0.293 against `F_op`**.
  2. The canonical gamma=0.5, rho=0.3 fixture is a NEAR-FOLD config outside
     this rung's domain (min image-delay gap 0.0336, so `xi_min >= 4` needs
     `w >= 317`). Its clean exponent is -2.377.

  Against the clean oracle, on the domain the rung actually serves:
  **median `d log err / d log w` = -3.010** (434 configs, w <= 60); -3.091 on
  the 4-image interior subset.

  This matches the DERIVATION rather than merely fitting: the leading omitted
  term is the `c3` term of the same stationary-phase series that produces the
  shipped `C1`/`C2`. The derivation was validated by reproducing
  `geometry._c1_polynomial` and `_c2_polynomial` to 2.4e-15 and 5.8e-14 over
  44 images spanning gamma 0.2-0.8. `c3` is purely imaginary and is a
  polynomial in the same `(prr, prt, ptt)` metric `geometry._saddle_metric`
  returns.

  ## The blocker was RIGHT, and the reason is bigger than expected

  Do NOT drop the `_uniform_error_estimate` leg on its own. **85% of the 434
  legs-1+2-passing configs have true ppGO error above 1e-4 somewhere in their
  band**, and one measured config (gamma=0.5, rho=0.4, theta=1.0) passes legs
  1+2 with a flat **1.078 absolute** error against `|F| ~ 0.2-1.9`.

  The cause is structural, and it also indicts legs 1+2:

  **`rho <= 1.0` admits caustic-EXTERIOR sources — 356 of 434 gate-passing
  configs have only 2 real images.** There raw ppGO omits the GHOST
  (complex-saddle) image entirely, a term absent from the 1/w series that
  decays as `exp(-w Im tau_c)`. Measured at gamma=0.9, rho=0.5, theta=1.5,
  w=60: raw ppGO error **5.81e-01**, ppGO+`ghost_kernel` **8.25e-03**, while
  the `w^-3` term is 1.60e-06 — **a pure `w^-3` certificate is 362,000x
  optimistic**. `geometry.ghost_kernel` already exists and is unconsumed.

  Worse, `_merging_fold_pair` computes `xi_min` from REAL images only, so for
  a 2-image exterior source it reports a large `xi_min` from the two
  well-separated real images while the actually-merging pair is the COMPLEX
  one it cannot see. Legs 1+2 are structurally blind there.

  ## The certificate

      ppgo_error_estimate(images, source, matrix, w_min) =
            sum_a  sqrt|mu_a| * |c3_a| / w_min**3
          + |ghost_kernel(w_min, source, matrix).kernel| * exp(-w_min*Im tau_c)

  evaluated at `w_min = float(dense_w.min())` (both terms decrease in `w`, so
  this is the sup over the band), refusing on `None`. Gate:

      est * PPGO_SAFETY <= CERTIFICATION_BAR,   PPGO_SAFETY = 10.0

  **It is an ESTIMATE, not a bound** — it bounds the leading omitted term, not
  the tail. Worst measured optimism 8.02x (gamma=0.9, rho=0.4, theta=0.35,
  w=17.9, where `c3` nearly cancels so the true error decays as `w^-4.4`).
  That optimism falls as `1/w` (3.06 by w=52), so it is a low-w artifact.
  Ratio table (6944 points): median 0.915, p90 1.004, p99 1.383, MAX 8.02.
  On the 4-image interior subset the MAX is 0.98 — never optimistic.

  Admitted-set accuracy where the engine can check it (w <= 60): max true
  error 3.15e-5 at SAFETY=10 against a 1e-4 bar; ZERO band-form exceedances at
  every safety factor. Above the DD ceiling (mpmath, w = 80/110/150) the ratio
  is 0.13-0.74, conservative at every point.

  Population effect (2496 sources, 8 gamma x 24 rho x 13 theta, kappa=0):

  | `w_min` | legs 1+2 alone | OLD 3-leg | NEW, S=10 |
  |---|---|---|---|
  | 60   | 1899 | 146  | 685 (4.7x old) |
  | 1e3  | 2394 | 602  | 2080 (3.5x old) |
  | 1e4  | 2475 | 1204 | 2398 (2.0x old) |

  Strictly wider than the old third leg everywhere, strictly narrower than
  dropping it. Cost per gate call (4 images): exact `c3` 6.27 ms. Do NOT
  substitute a cheap surrogate — five tried (`|C2|^1.5`, `|C1||C2|`, etc.)
  and every one under-predicts `|c3|` by 30x-300x somewhere. Either ship the
  exact routine or derive the closed-form `C3(prr, prt, ptt)` once.

  ## Acceptance, and what must be settled BEFORE shipping it

  1. **Separate the two `GhostDomainError` causes.** `ghost_kernel` raises the
     same exception for "four real roots, genuinely no ghost" (term = 0,
     correct) and "the continuation is degenerate on a principal axis" (MUST
     refuse). The measurement collapsed both to 0 and the second never
     occurred, so **that branch is untested**. Conflating them re-creates the
     F069 failure mode in a new place. This is the top item.
  2. **`PPGO_SAFETY = 10` has almost no headroom over the measured 8.02**, and
     the grid was kappa=0, beta=0, positive parity only. Either widen the
     measurement or raise the factor.
  3. **Absolute vs normalised bar.** The certificate is absolute on
     `|F - ppGO|`; the map's bar is `|F - ppGO| / max|F|` with `max|F| ~ 2-3`.
     Conservative UNLESS an admitted config has `max|F| < 1`.
  4. **`rho <= 1.0` remains mislabelled** as "interior". The ghost term covers
     the consequence, not the cause. Re-gauge or rename the leg.
  5. Extrapolation is validated to w <= 150 and used at `w_lo` = 1e3-1e5. The
     derivation supports the direction (term ratio `~|c4/c3|/w` shrinks; where
     `c3` cancels the estimate's own optimism shrinks as `1/w`), but no oracle
     in this repo reaches 1e3. Name it; do not hide it.
