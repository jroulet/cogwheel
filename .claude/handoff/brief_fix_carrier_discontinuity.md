# Build Brief: Fix CarrierDiscontinuityError in Training Pipeline

## Mission

The training pipeline (`train()` → `_train_band_charts` → `_reprovision_w_nodes`
→ `_build_farfield_chart` → `_farfield_box_to_smooth`) raises an unhandled
`CarrierDiscontinuityError` when a far-field exterior tile subtends a
degenerate caustic arc (span 0). The error message itself says "recorded as a
ladder-served gap" — suggesting this should be caught and skipped, not raised
as a fatal exception.

## Observed failure

```
CarrierDiscontinuityError: Far-field exterior tile subtends a degenerate
caustic arc (span 0); recorded as a ladder-served gap.
```

Stack: `train` → `_train_band_charts` → `_reprovision_w_nodes` → `_eps_for`
→ `_build_farfield_chart` → `_farfield_box_to_smooth`

## In scope

- Catch `CarrierDiscontinuityError` at the appropriate level
  (`_reprovision_w_nodes` or `_train_band_charts`) and treat it as a skip
  (the tile falls through to exact/ladder serving)
- Record the skip in the chart_reports (for the census to see)
- Tests verifying the degenerate-arc case is handled gracefully

## Out of scope

- Changing the tile proposal logic (the degenerate tile is a valid edge case)
- Training artifacts

## Constraints

- Fast tests only.
- Follow AGENTS.md and the spec/TODO workflow.
