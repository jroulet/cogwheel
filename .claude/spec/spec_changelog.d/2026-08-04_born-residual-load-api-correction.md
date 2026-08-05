---
bump: patch
---

### BornResidualChart.load API reference corrected in SPEC.md

Two locations in SPEC.md referenced `BornResidualChart.load(...)` as the
loading API for the shipped `born_residual_chart.npz` artifact, but no such
classmethod exists: `BornResidualChart` is a plain frozen dataclass with only
`covers` and `evaluate` methods.

Corrected both locations to reflect current reality:

- Engine row (table cell): "loads via `BornResidualChart.load(...)`" replaced
  with "is attached at construction time" plus a note that a `load` classmethod
  is not yet implemented.
- Conventions bullet: "Attaching it via
  `born_residual_chart=BornResidualChart.load(...)` completes" replaced with
  "Attaching a `BornResidualChart` instance (constructed from the shipped
  `.npz`) completes".

The corresponding DATA_CONTRACTS.yaml consumer entry removed in
`contracts_changelog.d/2026-08-04_born-residual-chart-consumer-fix.md`.
