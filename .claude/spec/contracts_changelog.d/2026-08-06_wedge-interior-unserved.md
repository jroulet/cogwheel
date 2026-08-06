---
bump: patch
---

### InteriorWedgeChart's coverage of the astroid interior softened to reflect measurement

`lens_amplification_surrogate`'s description named `InteriorWedgeChart` as
"the astroid interior's domain" as settled fact. `lensing_wedge_charts_
fail_the_eps_bar.md` (measured 2026-08-06, first completed production run on
the wedge path from `034fcf7`) found 0/12 wedge interior charts pass the
5e-2 eps bar (median eps 5.38e-1 vs the retired `ffin` path's 106/106 at
3.42e-4) -- the astroid interior is currently UNSERVED to tolerance and
falls through to the serving ladder.

Softened the `FarFieldChart` record sentence to call `InteriorWedgeChart`
the interior's *nominal* domain and note the current gate failure, with a
pointer to the todo fragment. Not a revert: the recommended fix (restore
`ffin`, per the same fragment) has not landed, so the contract still
describes `InteriorWedgeChart` as the wired path -- only the coverage claim
is corrected.
