---
section: Backlog
---

- **`_force_direct` does not reach the analytic intercepts — ratio/direct
  identity tests are trivially satisfied on intercepted anchors**
  `[housekeeping]` — observed 2026-08-18 (low-w gate-red fixture
  triage): none of the analytic intercepts (diffractive, and by
  inspection the sibling band-split rungs) honor the `_force_direct`
  escape hatch, so `test_lnlike_matches_direct_path_at_lattice_points`
  now passes vacuously on the 3 of 5 anchors the diffractive intercept
  serves (near_cusp, two_image, sheared_sw — measured; the fixture
  currently declines `_low_w_diffractive_serve` locally to keep its
  fork exercised). DECIDE: either thread `_force_direct` through every
  analytic intercept (one predicate at the dispatcher, not N copies) so
  ratio/direct identity tests exercise the real fork on every anchor,
  or re-anchor those tests to draws no intercept serves (fragile as
  rungs grow). The dispatcher-level predicate is the likely right home
  (one authoritative escape hatch). Small; pair with any upcoming
  likelihood-touching build.
