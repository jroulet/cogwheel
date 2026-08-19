# Inspector Short-Term Observations

## 2026-08-19 (diffractive_certificate_reach — review)

Scope: WP1 honest-ceiling up-bracket in `diffractive_w_low`; WP2 band threading
through 3 likelihood consumers; WP3 census mirror. Files:
`_diffractive.py`, `likelihood.py`, `serve_route_census.py`,
`test_lensing_diffractive.py`.

### Findings (INS-1-*)

- INS-1-001 (implementation/blocking): `_diffractive_bottom_ceiling` gained
  `(w_lo=None, w_hi=None)` and is now called with 3 POSITIONAL args at both
  nested-split production sites, but 4 probe lambdas in
  `test_lensing_born_certificate.py` (lines ~745, ~809, ~1525, ~1726) still
  bind the OLD 1-arg signature (`lambda lens: ...`). Result: 20 FAILED + 1
  ERROR in that file (all the same TypeError `takes 1 positional argument but
  3 were given`; one "zero comparisons" is a cascade). The file is NOT in the
  changed-file manifest, so the build's own gate never ran it. This is the
  GATE-CONTRACT-SWAP-CHANGES-ARITY laggard pattern: a signature change on a
  wrapper fails every un-updated probe silently-in-hindsight — sweep EVERY
  `_diffractive_bottom_ceiling=` / `diffractive_bottom_ceiling=` binding in
  `cogwheel/tests/`, not just the manifest's file. Fix = update lambdas to
  `lambda lens, w_lo=None, w_hi=None: ...`.

- INS-1-002 (implementation/over-certification): `_rootfind_w_high` assumes
  `relerr` (the honest N/2N tail ratio) is monotone non-decreasing in w, but
  it is NOT. Confirmed live at gamma=0.1, beta=0, y=(0.8,0.4): ratio breaches
  1e-4 near w~12.4 (1.19e-4), DIPS to 8.6e-5 over ~13.0-13.6, breaches again
  ~13.9. The doubling+bisection returns the LAST crossing (13.899), so the
  served band [w_lo, 13.9] CONTAINS the ~12.4 breach — over-certification.
  The test team ESCALATED it (documented in `CeilingTightnessTestCase`,
  `NONMONOTONE_DRAW` excluded from the sweep, `test_nonmonotone_tail_ratio_
  witness_gamma_0_1` pins it and flips red when the up-search is made
  first-breach-aware). NEW over-certification (the old candidate was
  conservative). Fix = make up-search first-breach-aware or add a
  non-monotonicity guard falling back to the conservative candidate.

- INS-1-003 (design/doc-vs-behavior): `_diffractive_bottom_ceiling` docstring
  claims whole-band-clear "collapses the host region to empty", but the
  nested composition (`band_split_low` strict-interior guard) treats
  `w_low == w_hi` as a NO-OP: `bottom_mask` empty, `host_mask == below_mask`
  (full). So when the honest ceiling reaches the band top, the analytic
  diffractive bottom (F_P) serves NOTHING in `_born_residual_analytic` — the
  up-bracket's reach-extension is nullified DISCONTINUOUSLY (F_P serves more
  as the ceiling rises, then drops to zero at w_hi). Confirmed: gamma=0.2
  unbounded ceiling 4.564, but w_hi=3.0 -> returns 3.0 -> bottom empty.
  `_low_w_diffractive_serve` handles w_low==w_hi correctly (below_low all-True
  -> F_P whole band). The test `test_band_split_byte_identity_when_whole_band_
  clear` pins the no-op behavior, contradicting the docstring. Direction
  (no-op vs collapse-host) is a triage call; at minimum the docstring must
  match the code.

### Note (not a finding)
- `_saddle_farfield_analytic` now calls `_diffractive_bottom_ceiling(lens, w_lo,
  w_hi)` but is gamma>1-only, where the method always returns None (parity
  wall) — the band-aware threading there is a harmless no-op.

### Tests run (fast tier)
- test_lensing_diffractive.py: 47 pass (64.8s).
- test_lensing_serve_route_census.py: 42 pass (23.5s).
- test_lensing_born_certificate.py: 20 FAILED + 1 ERROR (INS-1-001).
- born_residual_wiring / born_analytic_reachability / saddle_serve_gate: pass.
