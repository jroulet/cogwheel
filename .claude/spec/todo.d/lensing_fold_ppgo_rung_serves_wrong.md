---
section: Backlog
---

- **THE FOLD-PPGO INTERIOR RUNG IS WRONG BY ~21% WHERE IT SERVES, ON 791 LIVE
  CENSUS DRAWS — DECIDE WHETHER TO STRIP THE CORRECTION OR INVERT ITS GATE**
  `[→ spec]` — measured 2026-08-13 by a driver-commissioned investigation;
  the two supporting defects are recorded as [[FINDINGS F069]].

  This is a SERVING-CORRECTNESS item, not a test item. It needs a build, and
  the decision below is deliberately NOT taken here.

  ## What is wrong

  The rung (`likelihood.py`, the interior fold-ppGO block ~L1758-1805, mirrored
  in `surrogate_census.py` ~L464-493) serves when
  ``_uniform_error_estimate <= CERTIFICATION_BAR`` (1e-4). At the canonical
  served config it is wrong by **1.91e-1 median relative** (2.29e-1 absolute)
  while certifying **8.17e-5** — optimistic by **2581x**, and ~200x beyond the
  1e-3 level the `CERTIFICATION_BAR` comment already records as fatal to the
  0.05-nat lnL target.

  The certificate cannot be retuned into correctness: it is exactly
  ``(4/3) c_A / (w * delta_tau)``, which decays as 1/w, while the true error is
  w-INDEPENDENT (flat ~2e-1 from w=40 to w=2e5). Optimism therefore grows
  linearly in w, in exactly the direction the gate opens.

  The gate is also INVERTED. In closed form it serves iff
  ``w*delta_tau >= 13344*c_A`` — well-separated pairs far from the caustic,
  where the fold normal form is invalid. Measured at w=55, the fold correction
  beats raw ppGO ONLY for rho >= 0.93 (xi <~ 0.6); production demands xi >= 4.
  And ``c_A`` grows toward the caustic (1.0 at rho=0.3 -> ~2.6e2 at rho=0.99),
  making the gate HARDER to pass precisely where the correction is valid.

  ## The candidate fix, and why it is not applied here

  Recommendation from the investigation: stop applying the fold correction on
  this rung and serve raw ppGO (`operator.geometric_amplification`). Supporting
  measurement — raw ppGO vs the exact engine at the same fixture:

      w        40     60     70     100    150     200
      abs err  2.0e-4 5.1e-5 5.0e-5 8.7e-6 5.4e-6 5.3e-7

  decaying as ``w^-2.75`` (fit over w in [25,60] PREDICTED 5.5e-6 at w=150;
  5.4e-6 measured — three exact points beyond the fit range confirm it), so
  ~1e-7 or better at w ~ 5e4. F035 independently vindicates geometric optics at
  high w against GLoW.

  NOT APPLIED because it is a one-expression change to a live serving path
  affecting 791 draws, and it deserves a build with its own review: the census
  mirror must move with it, the `ppgo_fold` served-cause label's meaning
  changes, and the alternative (invert the gate to small xi / small eta, where
  the correction IS valid and lives entirely inside the engine's checkable
  domain) may be the better physics. Decide between them deliberately.

  ## Why nothing caught this

  1. The rung's served domain (w ~ 5e4) is far above the independent-oracle
     ceiling (w = 60 — NOT 150, see F069), so no test could compare it against
     truth.
  2. `fold_ppgo_correction`'s docstring asserts it "cannot make things worse
     than raw ppGO", which is false: at w=100 raw ppGO errs 8.7e-6 and the
     corrected value 2.1e-1, **24,000x worse than doing nothing**. That claim
     is why the rung has no error gate of its own.

  ## Acceptance

  Report the served error against an oracle valid at the SERVED w, not at a
  convenient w. If the decision is to keep the fold correction anywhere, its
  gate must be stated as a bound on the ERROR and demonstrated where the
  engine can check it.
