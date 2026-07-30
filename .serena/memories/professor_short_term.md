# Professor short-term (F054 caustic-reach closed form consultation, 2026-07-30)

Session task: Phase-1 domain review of the closed-form caustic reach replacing
the 720-point polar scan in `ppgo_map.caustic_geometry`. Answered 5 test-spec
questions (admissibility, direction, kappa, parity wall, tolerances).

## Observations worth consolidating into topic memories (Dreamer)

- **The reach reduces to a rational function of one variable `u`.** With
  lam=1-kappa, e=gamma/lam, |y|^2 = lam*P(u)/u^2 where
  P(u) = (1-u)^2(1+2u) + e^2(2u-1). The scan over 720 angles x 2 branches is
  computing the max of this over a 1-D interval; the extremum is a root of the
  cubic (u-1)(u^2+u+1-e^2)=0 plus endpoints. This belongs in
  `professor/microlensing_chang_refsdal` as the closed-form reach (companion to
  the already-recorded `saddle_caustic_max_y`/F026 macro-saddle fence and the
  An&Evans deltoid picture). The map is: u ranges over an interval bounded by
  the axis-cusp endpoints u=1-e (near cusp, the outer astroid cusp) and u=1+e;
  the sqrt-branch turnaround u=sqrt(e^2-1) only exists on the saddle.

- **SANITY anchor confirmed dimensionally consistent** with the code's own
  `p_i = (lam -+ gamma) - lam*u` convention in `caustic_derivatives` and
  W=lam(1-u), A=W-gamma, B=W+gamma. At u=1-e (kappa=0): reach=2*gamma/sqrt(1-gamma),
  =5.6921 at gamma=0.9 — matches SPEC.md cusp radius. The outer astroid cusp
  (the reach maximiser at positive parity) sits on the shear MINOR axis: at u=1-e
  the caustic point is on the B*sin(theta) axis => direction (0,+-1) in eigenframe.

- **Parity-wall singularity is the u->0 pole, reached as e->1 (kappa=0).** At e=1
  the near-cusp endpoint u=1-e -> 0, and |y|^2 = lam*P(u)/u^2 has P(0)=1-e^2=0 to
  first order but the 1/u^2 makes reach ~ 2*gamma/sqrt(|1-gamma|) blow up. gamma=1
  exactly = det A=0 = the named refusal already in critical_point/r_caustic. The
  closed form must inherit the SAME refusal test |gamma|==lam (and lam<=0), NOT
  compute a huge finite number. Confirms the incumbent's LensDomainError contract.

- **Direction quadrant is physically IRRELEVANT to the consumers.** Both
  call sites (test_lensing_ghost:577 `_anchor_source`, ppgo_map:881 `_measure_cell`)
  do `rho * reach * (R(angle) @ direction)` then feed the point into the engine,
  which is symmetric under the astroid/deltoid quadrant reflections (folding
  invariant, per priors_and_coordinates u1/u2 astroid fold). Only the AXIS
  ALIGNMENT and magnitude matter. Recommended canonical: eigenframe unit vector
  with a fixed sign convention (first non-negative component), and note the
  on-axis degenerate case (reach maximiser is on an axis => exactly (0,+-1) or
  (+-1,0)). This is a code-observation candidate for professor_code_observations
  too (the direction contract is looser than the incumbent's first-found value).

- **Tolerance reality check:** brief asks <=1e-9 rel vs an n_theta=11520 scan that
  itself only converges to ~1e-8 (measured at gamma=1.05). So the CLOSED FORM
  cannot be validated to 1e-9 AGAINST THE SCAN — the scan is the looser object.
  Correct test design: (a) closed-form-vs-scan at scan's own convergence floor
  (~3e-8 rel, generous 1e-7); (b) the STRICT 1e-9 gate belongs on the closed
  form's internal STATIONARITY self-check (d|y|^2/dtheta ~ 0 at returned point,
  analytic via caustic_derivatives) — that's machine-precision-able; (c)
  direction as an ANGLE agreement ~ scan angular resolution 2pi/11520 ~ 5.5e-4 rad.

## No code was written (Phase 1 plan-mode consultation).
