---
date: 2026-08-12
bump: patch
---

### Correct two SPEC.md claims that contradicted the code

Both surfaced by the Librarian during the `LobeExteriorChart` post-commit sync
and correctly left for someone who can adjudicate accuracy — the Librarian
owns SYNC, and the Inspector that would normally flag these is READ-ONLY.
Verified against the code, not against the other document.

**1. `_to_caustic_fixed` is additive on every exterior arm, both parities.**
SPEC.md claimed the coordinate was directional-MULTIPLICATIVE "on the astroid
interior, the astroid exterior arm, and the saddle interior arm". The code
(`surrogate.py::_to_caustic_fixed`) is multiplicative on the ASTROID INTERIOR
ARM ONLY (`|gamma| < 1` and `|y| <= r_caustic`); the astroid exterior uses
`rho = 1 + |y| - r_caustic`, and the macro saddle uses
`rho = 1 + |y| - _caustic_reach(gamma)` for every source regardless of side.
`DATA_CONTRACTS.yaml` already had this right ("additive exterior on both
parities ... multiplicative only on the astroid interior arm"), so the two
canonical surfaces contradicted each other.

Also noted: the saddle interior is not served through this coordinate at all —
it is charted lobe-locally by `LobeInteriorChart`/`LobeExteriorChart`. The
claim about a "saddle interior arm" of `_to_caustic_fixed` described a code
path that does not branch on side.

This matters beyond bookkeeping: the additive scalar form is exactly what made
the saddle corridor unrepresentable (a corridor source mapped to rho = -0.214
and `_from_caustic_fixed` raised), which is what the lobe-local exterior chart
was built to replace.

**2. `LobeInteriorChart.theta_to_u` is OPTIONAL, not required.**
SPEC.md said the loader "reads it unconditionally, so an absent map
hard-refuses (KeyError)". `_chart_from_npz` uses a soft `data.get` for
`lobe_interior`, `lobe_exterior` AND `exterior_polar`, deliberately: those
producers build the map only when given a `cusp_angle` and store `None` on the
raw-theta fallback, so a hard read would break the NPZ round-trip of a
legitimately map-less chart. ONLY the WEDGE (v3) hard-requires it, because the
wedge producer always builds one — there, an absent map means a stale
artifact.

No code changed. In both cases the code was correct and the spec was stale.
