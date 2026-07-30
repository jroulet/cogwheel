---
date: 2026-07-30
section: lensing
---

# Wedge-edge standoff deletion + analytic `_tube_normal` (build 1d)

Step 1d of `todo.d/lensing_caustic_relative_coordinates.md`'s "THE ANALYTIC
SWEEP", and the closing item (target 5) of `todo.d/lensing_analytic_derivatives.md`'s
estimator inventory. Driven by F044: the macro-saddle wedge edge is a REGULAR
point of the caustic (`y` and `dy/ds` both finite and nonzero in
`s = sqrt(theta_max - theta)`) — the `dtheta**-1/2` / `dtheta**-3/2`
divergences are artifacts of the `theta` parametrization alone, not evidence
of a singularity to stand off from.

Deletes `_WEDGE_EPS = 1e-3` from all 6 call sites in
`cogwheel/lensing/surrogate_training.py` with no inlined replacement, sampling
the macro-saddle wedge closed instead of excised. Rewrites
`surrogate_training._tube_normal` (the last numerical estimator on the
inventory in `todo.d/lensing_analytic_derivatives.md`) from a `dth = 1e-6`
forward difference of `critical_point` to the analytic tangent
`y' / |y'|` from `geometry.caustic_derivatives` — the F041 surface, so
`inward_sign` is the tripwire. Corrects four docstrings that called the wedge
edge a cusp (`geometry.caustic_derivatives`, plus three in
`surrogate_training.py`): the deltoid's three cusps are the interior
`|y'| = 0` roots; the wedge edge is a distinct regular point where the
`theta`-derivatives diverge only as a parametrization artifact.

Verified: `_WEDGE_EPS` gone with no inlined replacement (source-scanned);
`_lobe_winding_loop` closure gap exactly 0.0 (was 0.279 at gamma = 1.05, 9.3%
of lobe reach); no shrink in cusp/arc/reach/arc-span at any saddle band
(arc span slightly larger, as F044 predicted); `_tube_normal` analytic with
`inward_sign` unchanged on every production arc, pinned against a frozen
golden `inward_sign` table (Gate 4b, F041 flip-regression guard) and new
geometry tests comparing the analytic tangent to the retired forward
difference (Gate 4a). New Gate 5 (`WedgeEdgeServeRefusePredicateTestCase`)
pins the honest raise-or-diverge disjunction at the wedge edge for both
`critical_point` and `caustic_derivatives`. Two-lobe closure gap tightened
1e-2 -> 1.8e-3 (measured 1.670e-3) in `test_lensing_saddle_geometry.py`.

With this, `_tube_normal` was the last item on `lensing_analytic_derivatives.md`'s
numbered inventory (targets 1-4 were retired in builds 1b/1c); target 5 is
now DONE. `_branch_speed_profile` and `_find_cusps` remain in
`surrogate_training.py` but are no longer cusp-location estimators — 1b
repurposed them to size the cusp exclusion WINDOW only (an explicitly
deferred, still-open concern pending the F040 cusp-window schema build); they
are not a regression of this item.

Full suite green per Inspector PASS (no findings, no resolved_ids). SPEC.md's
microlensing-engine row already described the estimator retirement generically
("estimators `surrogate_training.py` retired in favor of the analytic
geometry cascade") from an earlier build and needed no further edit.

Steps 1e-9 of `lensing_caustic_relative_coordinates.md` remain open.
