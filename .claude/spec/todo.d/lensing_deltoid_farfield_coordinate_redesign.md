---
section: Backlog
---

- **DELTOID FAR-FIELD COORDINATE REDESIGN — the census's standing Q2
  verdict** `[→ spec]` — measured at production config post-F081
  (2026-08-15, tiling census): `redesign_needed = true`, reason "cusp ray
  strictly inside a tile angular span" (mis-allocation ratio 1.66, under
  the 2.5 bar — the cusp-ray condition alone triggers). A cusp ray inside
  a tile makes the 2/3-power directional reach non-monotone across it, so
  no per-tile node budget can fix the additive scalar-reach gauge; the
  tiling must be re-drawn in an eta-adapted per-lobe-edge coordinate (the
  log(rho-1) fix's deltoid analogue; SPEC already records the region as
  deliberate exact-engine fall-through "until a per-deltoid-edge
  coordinate design is certified"). Owed AFTER the 7a campaign (which
  excludes this region) and BEFORE 7b can claim table coverage there —
  7b's census will otherwise list the deltoid far-field draws as the
  known survivor set with this fragment as owner. Design inputs: the
  Q2 verdict machinery in `tiling_census.py` (re-run per candidate
  coordinate — a candidate passes when no tile contains a cusp ray and
  the mis-allocation ratio stays under the bar), the F074 Pearcey
  controls for the near-cusp annuli, and the per-lobe D2 fold (one lobe
  edge fundamental, mirrors served by the gauge-image law).
