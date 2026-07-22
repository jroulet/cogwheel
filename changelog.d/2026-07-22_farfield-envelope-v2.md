---
date: 2026-07-22
---
### Far-field envelope redefinition and serving mirror (Build 8g-b)

Far-field surrogate charts now fit the full post-geometric-optics
remainder ``E_ff = F − Σ_real H_a e^{i w τ_a}`` — the criticality
switch is forced on for every real image and the nearest-caustic
demodulation carrier is dropped, removing the lobe-assignment
discontinuity that made diagonal-straddling tiles unfittable. The
serving path mirrors the definition via a required per-chart
``envelope_definition`` tag; loading a far-field chart without the tag
(pre-8g-b artifacts) is refused with a rebuild instruction. The
far-field gate currency is now F-normalized (``farfield_eps_max``
1e-3). Tube charts are unchanged.
