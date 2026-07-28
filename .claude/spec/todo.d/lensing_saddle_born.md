---
section: Backlog
---

- **Born rung for the MACRO-SADDLE (`gamma > 1`) is unwritten and was
  untracked** — `chang_refsdal/_born.py` is positive-parity ONLY, by
  construction, not by omission:

  * the expansion origin is `sqrt(mu_macro)` with
    `mu_macro = 1 / ((1 - kappa)**2 - gamma**2)`, whose radicand is POSITIVE
    only below the parity wall. At `gamma > 1` it flips sign, so the origin is
    imaginary and the macro image is a saddle carrying a Morse phase — the
    series must be expanded about `sqrt(|mu_macro|) * exp(-i*pi/2)`, not about
    a real amplitude;
  * guard B (`_born.py` ~L78) is documented as a "positive-parity convergence
    margin" on `gamma_p = |gamma| / (1 - kappa)` approaching 1 FROM BELOW; at
    `gamma > 1` the relevant degeneration is different and that guard does not
    apply;
  * the docstring already states the caller "must have gated the parity wall",
    so the module is honest about its scope — it simply has no saddle branch.

  Consequence for coverage: the far annulus `3.0 < |y| <= 4.2426` is where the
  Born rung is meant to buy zero-quadrature serving. At `gamma > 1` that
  annulus has NO Born rung at all, so even once `b1` lands
  ([[lensing_born_b1_derivation]]) the saddle half of the annulus still falls
  through to the exact engine. Correct and certifiable
  (`w * |y| <= 60` never binds inside the prior), but NOT zero-quadrature.

  ORDERING (owner-directed): this follows the positive-parity `b1` derivation,
  not the other way round. Deriving `b1` first fixes the O(1) numerator in the
  regime where the reference (`operator.F_op`) is cleanest; the saddle version
  then reuses that structure with the Morse-phase origin rather than
  re-deriving from scratch.

  Owed, in order:
  1. `b1` closed form for positive parity (the existing TODO).
  2. Saddle expansion origin + its own convergence guard, with the parity wall
     `gamma = 1` excluded as the measure-zero named refusal it already is.
  3. A saddle accuracy gate against `operator.F_op` mirroring the T1 target
     (rel err < 1e-3), NOT a re-use of the positive-parity fixture.
  4. Only then wire the saddle branch through the fact-4 slot in
     `likelihood.py::_surrogate_coefficients`.

  Recorded 2026-07-28 after the owner noticed it was missing from the plan
  list. It had never been written down: the only Born fragment was the
  positive-parity `b1` derivation, and "saddle Born" existed solely as a
  spoken step in the build ladder.
