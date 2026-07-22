---
section: In progress
---
- **Far-field tiling + eps registration gate + saddle-tube tail** `[→ spec]` —
  the full-box campaign (2026-07-22, 86 charts / 54 MB) exposed that the
  far-field stage of `surrogate_training._train_band_charts` is fixture-scale
  legacy: every far-field chart is placed at the SAME hard-coded
  `box_center = (caustic_reach + eta_max + 0.2, 0.0)` (halfwidth 0.15);
  `max_farfield_regions` builds duplicates. End-to-end census against the
  trained artifact: 1/1024 prior draws served (942 out-of-box). Build 8g:
  (1) replace the single-box placement with a mass-stratified tiling of the
  prior's shear-frame y-support (`Y(m) = min(307/m, 3)`, whole-band w
  containment per stratum, w-caps per parity per F019); (2) add an
  eps-based registration gate (`heldout_eps` above a config bar -> chart
  recorded but NOT registered/served; no gate exists today); (3) diagnose
  and fix the saddle tube tail (3 charts at 0.43-2.15 max-normalized
  envelope error, ~5 more >= 0.09; recurring arcs tube_2/tube_5).
