---
section: Backlog
depends_on: [2026-08-10_exterior_rho_phase_carrier]
---

- **Exterior fold-carrier phase demodulation (recover ghost-transition zone)**
  `[→ spec]` — identified 2026-08-10 after the ghost-gate exclusion build.

  The ghost-gate exclusion (cf81d66) correctly excludes tiles where the
  unsubtracted ghost dominates the KERNEL_SUM residual, but that is ~40%
  of the exterior prior box (measured). Those draws fall to the exact
  engine (correct but ~tens of ms/node vs surrogate speed). This build
  recovers them at surrogate speed via analytic FOLD-CARRIER phase
  demodulation:

  The ghost's phase oscillation is e^{iw·Re(tau_c(rho))}, where tau_c is
  the delay of the fold point where the two real images merged (the ghost
  is the single Fresnel/Airy blob centered there). Demodulating the
  envelope by e^{-iw·Re(tau_c(rho))} per node before fitting flattens the
  rho-phase winding (measured: 16.7 -> 3.2 rad over rho in [1.3, 2.1]),
  leaving a smooth splineable residual — WITHOUT needing the decay gate
  (Re(tau_c) is well-defined regardless of Im(tau_c)). tau_c comes from
  `geom.ghost_kernel(...).delay`, already computable.

  IMPORTANT (measured): do NOT demodulate by the full complex tau_c
  (multiply by e^{+w·Im}) — that divides out the ghost's decay and
  amplifies everything by ~19x at w=30 (numerically explosive). Only the
  phase (Re) is demodulated; the ghost's smooth amplitude decay e^{-w·Im}
  is left in the residual for the spline to fit (it is monotone, not
  oscillatory). Equivalently, where the ghost model is accurate
  (Im tau_c >= 0.4) the MINUS_GHOST label subtracts analytically and the
  reconstruction re-adds it at serve — the layered strategy:
  (1) MINUS_GHOST subtraction where the model is accurate,
  (2) fold-carrier phase demodulation in the transition zone,
  (3) exclusion -> exact engine for the residual.

  ACCEPTANCE: the exterior probe produces ~70 charts with all held-out eps
  under the 1e-3 bar, AND the previously-ghost-excluded region (~40%) is
  now served by the surrogate (census shows the ghost-region draws served,
  not falling to the engine); round-trip to machine precision.
