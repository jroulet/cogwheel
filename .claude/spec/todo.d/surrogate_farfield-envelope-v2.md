---
section: In progress
---
- **Far-field envelope redefinition + consolidated tiling (Build 8g-b)**
  `[→ spec]` — mid-campaign probe (2026-07-22) found far-field tiles
  straddling the astroid diagonals fit garbage (eps 0.3-0.9): the
  criticality switch keyed on nearest-caustic ``tau_c`` flips lobes on the
  equidistance line, leaving a resolved image un-subtracted (envelope
  jumps x1500 mid-tile, measured). Fix: far-field charts subtract the
  FULL ppGO sum (switch forced on) with no ``tau_c`` carrier — envelope
  becomes smooth ~1e-4 ``F − Σ H_a e^{i w tau_a}``; serving reconstruction
  mirrors the new definition via a chart-meta envelope-definition tag;
  tiling re-provisioned coarser against the now-tiny envelope. Tubes
  unchanged (their carrier/switch are correct near-caustic).
