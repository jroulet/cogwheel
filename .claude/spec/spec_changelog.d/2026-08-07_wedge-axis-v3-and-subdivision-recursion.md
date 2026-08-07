---
date: 2026-08-07
bump: minor
---

### Wedge axis schema v3, bounded subdivision recursion, exact-parametrisation `r_caustic`

Three changes to the interior-chart and caustic-geometry layer.

**Wedge axis schema v3.** The `InteriorWedgeChart` paragraph now states the
shipping contract: `axis_schema = 'wedge_caustic_relative_v3'` as the only
known tag, `theta_to_u` / `u_grid` field names, and `theta_to_u` REQUIRED with
a hard refusal on an absent or stale map. Arc-length names stay with the
charts that genuinely hold arc length (tube, lobe-interior, far-field); both
validators delegate to a shared `_validate_axis_map` core with no length-scale
bound, because `u` is `rad**(2/3)`.

`_wedge_cusp_axis_map` now HARD-RAISES for a `theta` bound outside
`[0, pi/2]`, the D2-folded fundamental domain, instead of returning a silently
complex array from the negative base of `(pi/2 - theta)**(2/3)` — a failure
that previously surfaced frames later as an unrelated-looking cast error
inside `np.interp`. A clamp was rejected: a bound outside the domain can only
come from a caller that failed to fold, and clamping would serve the reflected
tile's basin and mask the fold bug.

**Bounded subdivision recursion.** Both tile subdividers were single-level: a
child that still failed the eps bar became a ladder-served gap rather than
being split again. They now share one generic `_subdivide_tile` parameterised
by splitter / builder / gate / admission predicate, with
`MAX_SUBDIVISION_DEPTH = 3`, and each reports its achieved depth so a runaway
is visible. `_subdivide_farfield_tile` and `_subdivide_wedge_tile` remain thin
wrappers, additive report keys only, pinned byte-identical at depth 1.

Motivation, measured on the astroid interior at `gamma_mid = 0.495`: 13/16
children cleared at one halving and the three that did not were marginal —
6.50e-2, 6.70e-2, 5.95e-2 against a 5e-2 bar — while each halving had been
buying 2-5x.

**`r_caustic`.** Bracket density for the positive-parity astroid is now fixed
internally at 48 (the macro saddle keeps 720, where coarse uniform bracketing
misses deltoid lobe entry/exit), and `n_sample` is an accepted-but-ignored
deprecated kwarg. Measured 1.788 s -> 0.169 s for 200 calls (10.6x). Driver
cross-tree verification over 6080 (gamma, theta) samples spanning
`gamma in [0.05, 0.95]` with near-cusp refinement: zero refusal mismatches,
worst 7.59e-15 relative — `brentq` convergence noise, not a value change.

NOTE on the build brief that commissioned this: it asserted `r_caustic` erred
0.32% at `gamma = 0.9, theta = pi/2` (5.67376 vs 5.69210) and scoped a
branch-selection fix around it. That premise was STALE — the fix had already
landed, and both trees return 5.692099788303084 bit-for-bit. The delivered
change is the speedup alone. See
`.claude/spec/todo.d/lensing_brief_premises_are_unverified.md`.
