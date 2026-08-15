---
date: 2026-08-15
bump: minor
---

Both parities now train one representative arc per D2 gauge orbit: the
astroid's four arcs already collapsed to the single pi/4-bracketing arc
(4 -> 1), and the saddle's six detected deltoid arcs now collapse to one
representative per orbit derived from the fold law (typically 6 -> 3).
The `max_tube_arcs` config knob is retired. F081's starvation fix rides
along: the per-band scalar sizing saddle lobe admissions and the deltoid
far-field inner edge is now the narrowest per-arc tube shell
(`min_eta_max`) rather than the widest (`max_eta_max`), which had
starved every saddle lobe admission and the far-field inner edge alike.
`max_eta_max` still sizes the tube w-grid cap and the astroid
interior-skip/wedge extent. Mirrored into `tiling_census.py`,
`scripts/census_dry_run.py`, `scripts/train_surrogate_production.py`.
Build `saddle_tube_fundamental_training`.
