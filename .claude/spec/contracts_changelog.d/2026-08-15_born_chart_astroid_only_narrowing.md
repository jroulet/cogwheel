---
date: 2026-08-15
bump: patch
---

`born_residual_chart`: the `wire_serving_artifacts` build (5a739b6) claimed
in its commit message and `completed.d/2026-08-14_lensing_wire_serving_
artifacts.md` that in-build escalation INS-3 ("both parities" claim vs the
astroid-only shipped artifact) was "fixed by text-narrowing" — but the
actual diff left the description's "covering the far exterior (rho > 2)
on both parities" clause, and the gate prose's "(far exterior, both
parities)", byte-for-byte unchanged; the claimed fix never landed. Post-
commit verification (Inspector findings INS-1-002/INS-2-002/INS-3-001,
carried in `.claude/sync_issues.json`) confirmed the shipped
`cogwheel/data/born_residual_chart.npz` has `gamma_grid` entirely below
1.0 (astroid parity only; `rho_grid` >= 2.0, `log_w_grid` spanning w in
[5, 60]) with no saddle node — `covers()` refuses any `gamma > 0.9` query
regardless of code-level "both parities" phrasing. Narrowed both
occurrences to the shipped truth (astroid-only; saddle Born nodes are a
training-campaign decision, not a shipped capability) and corrected the
gate prose from `covers(gamma, rho)` to the full coded guard
`covers(gamma, rho, chart_w)` (box containment plus the trained log-w
band, refusing rather than cubic-extrapolating). Mirrored the same
narrowing into the `_born_residual_analytic` docstring in
`cogwheel/lensing/likelihood.py`, which repeated the identical stale
"(exterior, both parities)" / two-argument `covers()` phrasing.
