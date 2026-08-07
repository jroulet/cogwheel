---
date: 2026-08-07
section: Backlog
---

### Bounded subdivision recursion, wedge axis v3, and a 10.6x `r_caustic`

Closes three `todo.d` fragments, shipped by the `subdivision_recursion` build
(`bca9534`) with driver post-build verification.

**`lensing_subdividers_are_single_level` — done.** Both tile subdividers now
share one generic `_subdivide_tile` parameterised by splitter / builder / gate
/ admission predicate, with `MAX_SUBDIVISION_DEPTH = 3` and an achieved depth
reported per tile so a runaway is visible. `_subdivide_farfield_tile` and
`_subdivide_wedge_tile` remain thin wrappers with additive report keys only,
pinned byte-identical at depth 1. The two subdividers were unified rather than
each gaining its own recursion — they were duplicates before the change.

**`lensing_wedge_u_map_stored_in_arclength_fields` — done.** Wedge-only rename
to `theta_to_u` / `u_grid`, schema `wedge_caustic_relative_v2` -> `v3` with the
map REQUIRED and stale artifacts hard-refusing. Arc-length names stay with the
tube, lobe-interior and far-field charts that genuinely hold arc length; both
validators delegate to a shared `_validate_axis_map` core with no length-scale
bound, since `u` is `rad**(2/3)`. `_wedge_cusp_axis_map` now hard-raises
outside `[0, pi/2]` instead of returning a silently complex array.

**`lensing_r_caustic_should_root_find_not_scan` — closed, PREMISE CORRECTED.**
The fragment (and the brief built on it) claimed `r_caustic` inverted the
parametrisation by scanning, returning 5.67376 against an exact 5.69210 — a
0.32% error. That was WRONG at the time the build ran: the function already
brentq-refined each bracket, the branch-selection fix had landed earlier, and
both the pre-build and post-build trees return `5.692099788303084`
bit-for-bit. There was no accuracy defect to fix.

What shipped is the speedup alone: the positive-parity bracket count is fixed
internally at 48 (the macro saddle keeps 720, where coarse uniform bracketing
misses deltoid lobe entry/exit), and `n_sample` becomes an accepted-but-ignored
deprecated kwarg. Measured 1.788 s -> 0.169 s for 200 calls (10.6x). Driver
cross-tree verification over 6080 `(gamma, theta)` samples spanning
`gamma in [0.05, 0.95]` with near-cusp refinement: zero refusal mismatches,
worst 7.59e-15 relative — `brentq` convergence noise, not a value change.

That noise moved a bits-exact golden by ONE ULP and turned the tree gate red,
which is how the stale premise was found at all. Both process failures are
tracked: [[lensing_brief_premises_are_unverified]] and
[[lensing_golden_fixture_recomputes_geometry]].

**Not closed by this build.** The interior is still UNSERVED to tolerance at
production scale — recursion makes the marginal tiles reachable but no
production training run has been made since. Wedge charts from the 2026-08-06
probes are unloadable under v3 and must be retrained before any measurement
built on them is trusted.
