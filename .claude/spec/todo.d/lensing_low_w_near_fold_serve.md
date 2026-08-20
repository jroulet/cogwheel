---
section: Backlog
---

- **LOW-W NEAR-FOLD ANALYTIC SERVE — the near-fold shell has no dedicated
  serve (owner ruling 2026-08-20, DO NEXT — the fence build has LANDED)**
  `[→ spec]` — the near-fold band of the diffractive rung has NO analytic
  serve at low w. The fence build (diffractive_certificate_fit_fenced +
  interior-fix + gamma-fence, SHIPPED: near-fold shell declines, deep
  interior served by the calibrated fit, wall band gamma'>0.5 routes to
  Schwinger) declines draws with `rho = |y'|/|y_c(theta)|` in `[RHO_LO ~
  0.6, 1+DELTA ~ 1.4]` (returning None) so they fall through to the serving
  ladder. BUT the uniform Airy fold arm (`_airy_fold.fold_amplification`)
  REFUSES at all w in the diffractive band (measured at the fence corner
  gamma=0.41, r=0.55: refuses at w=0.5..10): its control `xi = (3 w
  Delta_tau / 4)^(2/3)` is a large-w asymptotic, and the arm is only OFFERED
  in the ladder for w > 60 (the DD band w <= 60 is unconditional exact-wave
  `f_schwinger`). So the fenced near-fold shell routes to the EXACT ENGINE,
  not the fold arm. Measured: the shell (rho 0.9-1.2) is 332/4462 = 7.4% of
  engine-residual demand (demand_census_post_born_10k.json). This is
  correct-but-expensive and leaves the low-w near-fold band without a
  dedicated analytic rung. The WALL band (gamma'>0.5) is SEPARATE and is
  RESOLVED by routing to Schwinger (owner ruling: the order-16 series has a
  convergence-radius collapse there — a square-root branch point not
  representable at any practical order — so Schwinger is the correct serve,
  and a fold-adapted serve is the right target for the NEAR-FOLD shell, not
  the wall).

  ORDERING CONSTRAINT (owner): this MUST land before any demand-census-
  driven work (the tiling-plan refresh / campaign sizing), because the
  engine-residual number changes when the shell stops being engine-served.
  The fence build itself is the prerequisite (it defines the shell geometry
  and the `_caustic_rho` discriminator).

  DESIGN QUESTIONS (Professor to rule):
  - Is a physically sound low-w near-fold serve realizable? The diffractive
    series (weak-deflection, shear-operator expansion) is wrong near the
    fold; the fold arm (large-xi asymptotic) is wrong at low w. At small w
    the wave field near the fold is SMOOTH (long wavelength resolves no
    caustic structure), so the amplification should be close to the
    geometric/fold-limit value with O(w^2)-ish corrections — suggesting the
    low-w near-fold serve may be a smooth function of (w, distance-to-fold)
    rather than a hard expansion. If so, is it:
    (a) a low-w uniform-Airy-CORRECTED diffractive series (two-image beat
        near the fold, like the tube charts' two-carrier F_ref);
    (b) a low-w fold-adapted expansion in distance-to-caustic;
    (c) or is the 7.4% shell small enough that fence-to-engine is the right
        engineering call and the serve is not worth building?
  - The fence discriminator rho_code is monotone-but-miscalibrated
    (`caustic_point` parametrizes by critical-curve angle phi != theta, up
    to 1.75x under-estimate of distance-to-fold); any new serve must use the
    same O(1)/engine-free discriminator or a cheap correction, never a
    per-draw numerical root-find.
  - MUST be validated against the exact engine (sup-over-w, CERTIFICATION_BAR
    = 1e-4) and must never over-serve (the fence's whole point).

  ACCEPTANCE: the near-fold low-w band is served analytically at <= 1e-4
  against the exact engine (or the shell is explicitly ruled engine
  territory with measured cost), and the serve-route census no longer counts
  the shell as engine demand. Then the tiling-plan refresh and campaign
  sizing can proceed on the honest post-serve demand map.
