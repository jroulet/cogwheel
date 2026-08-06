---
date: 2026-08-06
section: Backlog
---

### InteriorWedgeChart wired into training — the ffin path retired

Commit `034fcf7` closes `todo.d/lensing_interior_wedge_chart_unwired.md`
(audited 2026-08-06 in `aac4d16`). The astroid interior is now trained as
`InteriorWedgeChart` in wedge-fixed `(r, theta_wedge)` coordinates: new
helpers `_wedge_interior_tiles` and `_build_wedge_chart` in
`cogwheel/lensing/surrogate_training.py` call the pre-existing
`InteriorWedgeChart.from_wedge_engine` (`surrogate.py`), transcribing the
already-wired macro-saddle `LobeInteriorChart` path per the fragment's own
template.

The `ffin` path is deleted, not left reachable: `_farfield_interior_tiles`
is gone, along with the `definition=INTERIOR_SACR_C` branch of
`_build_farfield_chart` (its keyword-only `definition` parameter is
removed from the signature), the `region == 'interior'` branch of
`_subdivide_farfield_tile`, and the interior `FarFieldChart` branch of
`_heldout_eps`. `_interior_admission` is unchanged -- it is the live
exterior-tiler dependency, not interior-only as the original brief assumed.

The `INTERIOR_SACR_C` envelope label itself is unchanged and still carried
by both `LobeInteriorChart` and `InteriorWedgeChart`; only its pairing with
`FarFieldChart` retired. DATA_CONTRACTS.yaml's `lens_amplification_surrogate`
description corrected to match (`contracts_changelog.d/
2026-08-06_farfield-interior-retired.md`).

DIVISION OF LABOUR after this build, per the closed fragment's own framing:
astroid interior -> `InteriorWedgeChart`; saddle lobe interiors ->
`LobeInteriorChart`; exterior both parities -> `FarFieldChart`; near-caustic
-> `TubeChart`.

Two items spun out of this build and remain open, tracked separately:
`todo.d/lensing_gated_training_suite_is_vacuous.md` (pre-existing, measured
by A/B, unrelated to this change) and `todo.d/lensing_serving_ladder_guards_
are_red.md` / `todo.d/lensing_fast_tier_hangs_in_mpmath.md` (the tree-wide
fast gate could not complete, so this build's own full-suite tally is
unverified -- GATE NOT VERIFIED per the commit message).
