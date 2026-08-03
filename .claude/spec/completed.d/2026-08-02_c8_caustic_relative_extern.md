---
date: 2026-08-02
section: Backlog
---

C8 (far zone becomes caustic-relative; annulus retired) from
lensing_caustic_relative_coordinates, step 5.
ANNULUS_INNER_RADIUS, GAMMA_FENCE = 3/4, and saddle_caustic_max_y() all
deleted from _born.py. born_gate reduced from three guards to two (guard B:
parity-wall margin; guard A: band split). surrogate_census.classify_fallthrough
now uses rho > 1 (exterior-to-caustic) for the born category; constant
_BORN_ANNULUS_OUTER_RADIUS deleted. ppgo_map.annulus_rho renamed caustic_rho.
SPEC.md updated to reflect exterior region, two guards, and rho > 1 criterion.
