---
section: Backlog
---

- **`r_caustic` scans 720 points to invert a smooth monotone map — root-find
  instead** `[housekeeping]` — measured 2026-08-06.

  `r_caustic(gamma, theta, *, kappa=0.0, n_sample=720)` returns the distance
  from the origin to the caustic along a SOURCE-plane direction. The caustic
  itself is available EXACTLY and analytically from `critical_point(gamma,
  theta_lens)`, but that parametrises by the LENS-plane angle, and
  `phi(theta_lens)` has no closed-form inverse — hence the scan.

  Scanning 720 points and interpolating is the wrong way to invert a smooth
  monotone map. A `brentq` root-find on `phi(theta_lens) = phi_target` reaches
  machine precision in ~10-15 exact `critical_point` evaluations.

  ## Cost of the status quo

  - ACCURACY: `r_caustic(0.9, pi/2) = 5.67376` against the exact
    `|critical_point(0.9, pi/2).source| = 5.69210` — **0.32% error**, growing
    with gamma as the cusps sharpen (agrees to 5 decimals at gamma=0.3). This
    propagates straight into the wedge chart's radial coordinate, which is
    `r = |y| / r_caustic(gamma, theta)`.
  - SPEED: 200 evaluations take 1.85 s. Because `r_caustic` is called per grid
    point in several places, this dominated a driver probe badly enough to
    need rewriting around a precomputed table (a 260x260 grid implies ~48M
    inner evaluations).

  ## A closed form was tried and is WRONG — do not retry it

  The Chang-Refsdal caustic is NOT the algebraic astroid
  `(x/A)^(2/3) + (y/B)^(2/3) = 1` with `A`, `B` the cusp radii. Fitted at the
  axes (where it is exact by construction) it errs in between:

      gamma = 0.2   0.5%
      gamma = 0.495 3.5%
      gamma = 0.9   21%

  That form is a low-gamma approximation that fails as the cusps sharpen.

  ## Exact structure that DOES hold, and is worth deriving properly

  `r_caustic(gamma, theta_waist) == gamma` EXACTLY at every gamma tested
  (0.200, 0.300, 0.495, 0.700, 0.900 — dead on), where
  `theta_waist = argmin_theta r_caustic`. The rejected astroid approximation
  does NOT reproduce it (0.201, 0.304, 0.512, 0.756, 1.062), so this is a real
  property of the exact curve. Use it as a test oracle; it also suggests more
  closed-form structure is available than the driver found by guessing.

  ## Work

  - Replace the scan with a `brentq` inversion of the exact parametrisation;
    keep the signature, drop `n_sample` (or keep it as an ignored deprecated
    kwarg if callers pass it).
  - Pin with the waist oracle above and against `critical_point` at the axes.
  - MUST NOT land while a build is measuring eps against the wedge coordinate:
    changing this normaliser moves `r = |y|/r_caustic` under any in-flight eps
    acceptance. Sequence it after
    [[lensing_wedge_angular_axis_is_cusp_singular]] lands.

  ACCEPTANCE: `r_caustic` agrees with `|critical_point(...).source|` to machine
  precision at the axes and satisfies the waist relation to ~1e-12; a 200-call
  benchmark is at least an order of magnitude faster than the 1.85 s scan.
