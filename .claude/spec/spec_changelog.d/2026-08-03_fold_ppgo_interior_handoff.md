---
bump: patch
---

### Fold-ppGO interior handoff serve path

SPEC.md "Microlensed waveform & likelihood" row updated to describe the new
fold-ppGO interior handoff path in `_surrogate_coefficients`: interior positive-parity
draws above the `InteriorWedgeChart` w-ceiling are now served when `xi_min >= 4.0`
and the uniform error estimate passes `CERTIFICATION_BAR`. Census breakdown updated
from 6-way to 7-way (new category: `ppgo_fold`).
