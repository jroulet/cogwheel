---
section: Backlog
---

- **TRIM SADDLE TUBE TRAINING TO A FUNDAMENTAL ARC SET — before the
  campaign pays for symmetry-redundant charts** `[housekeeping]` — the
  tube_d2_fold build (2026-08-14) serves tube queries through a D2
  gauge-image search (`_tube_theta_inframe`: the four images theta,
  pi-theta, -theta, pi+theta, identity first, first one inside the
  chart's frame serves), so a trained arc's three D2 mirrors serve
  through it. Astroid training already exploits this (one pi/4-bracketing
  arc, 4 -> 1). Saddle training still builds `arcs[:max_tube_arcs]`
  (production knob 20 = all 6 deltoid arcs), so up to half the saddle
  tube charts duplicate what their mirror images already serve —
  symmetry-redundant training cost the owner's directive forbids.
  Derive the saddle fundamental arc set FROM THE IMAGE SEARCH (which arcs
  are NOT D2 images of another trained arc — expect 6 -> ~3; note the
  orientation-reversing gauge<->source map measured in
  `_tube_training_arcs`' comment, and that near-cusp gauge slivers span
  arcs, so verify serve coverage over the full ring with the image
  search active, not by arc bookkeeping alone), then restrict the saddle
  branch the way the astroid branch already is, with the same
  detected-vs-trained count split the astroid test pins. MUST land
  before the training campaign (7a); the tiling census (6/7) counts
  charts against the live serving design and will independently flag the
  redundancy if skipped.
