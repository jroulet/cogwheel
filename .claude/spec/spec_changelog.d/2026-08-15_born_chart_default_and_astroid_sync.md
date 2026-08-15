---
date: 2026-08-15
bump: patch
---

The `wire_serving_artifacts` build (5a739b6) updated the pipeline-row
sentence for the Born intercept correctly but left an older paragraph
below it stale: it still said the chart defaults to `None` (opt-out) and
described only the legacy fact-4 `_surrogate_coefficients` slot, with no
mention of the new auto-attach default or the first-class
`_born_residual_analytic` intercept that now shares the same attached
`BornResidualChart`. Rewrote the paragraph to state the auto-attach
default (`_AUTO_BORN_CHART` sentinel, explicit-`None` opt-out), that both
the legacy surrogate slot and the new intercept consult the same attached
chart, and that the shipped artifact covers the astroid parity only
(mirrors the `contracts_changelog.d/2026-08-15_born_chart_astroid_only_
narrowing.md` DATA_CONTRACTS.yaml fix for the same artifact).
