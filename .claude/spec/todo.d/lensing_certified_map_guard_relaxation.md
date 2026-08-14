---
section: Backlog
depends_on: [lensing_wire_serving_artifacts]
---

- **NEXT-SESSION ORDER 5/7 — RE-VALIDATE AND SELECTIVELY RELAX THE
  CERTIFIED MAP'S SADDLE rho<1 GUARD** `[→ spec]` — the artifact HOLDS
  certified saddle rho<0.5 cells (w_cert 27.7/19.2/15.9 by gamma band;
  independently reproduced at 16-28) but `CertifiedPpgoMap.w_cert`
  hard-refuses all saddle rho<1 as F073-era defense-in-depth.

  RE-VALIDATED 2026-08-14 (driver pilot, F080 — "verify, do not assume"
  was the right call): the three certified cells are NOT equal.
  - gamma [1.157, 1.339]: CLEAN (5/5 configs, sup 8.7e-5) — eligible for
    per-cell relaxation now.
  - gamma [1.339, 1.550]: MARGINAL (1.0-1.4e-4 at the w_cert node only,
    under bar by w=27.7) — relax only with a small w_cert raise or after
    re-measurement; decide from a denser scan of that node.
  - gamma [1.100, 1.157]: CONTAMINATED — sup err 4.49e-1 (3.5 orders over
    bar) at the lower-gamma-edge x transverse-angle corner; would not
    re-certify at its own center today (fan-worst 1.21e-4). STAYS REFUSED
    until the 7a retrain re-measures it with edge-biased, worst-over-cell
    sampling (the F080 sampling-blindness fix).

  Scope of the build therefore: per-cell relaxation keyed on re-validation
  evidence (a stored or asserted re-validation stamp, not a blanket parity
  x rho predicate), serving the clean cell(s) as the second certification
  layer; the F073 deltoid-straddling-annulus reasoning stays in force for
  everything unvalidated. The F080 fan-asymmetry question (mirrored fan
  angles disagree 2.4x under exact D2 symmetry) must be answered before
  the retrain trusts the fan — route it to the Professor in this build or
  the 7a brief, whichever runs first. Full re-validation is cheap (~60 s
  training fidelity, ~10 min corner-refinement, measured).
