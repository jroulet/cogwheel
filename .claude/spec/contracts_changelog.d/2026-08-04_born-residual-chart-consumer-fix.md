---
bump: patch
---

### Remove non-existent BornResidualChart.load consumer entry

The previous librarian sync incorrectly added
`cogwheel/lensing/born_residual_chart.py::BornResidualChart.load` as a
consumer of the `born_residual_chart` artifact.  No such method exists:
`BornResidualChart` is a plain frozen dataclass with `covers` and `evaluate`
methods only.  The `.npz` is written by `scripts/train_born_residual.py`
via `np.savez` and read by callers who construct the dataclass manually.

The only legitimate production consumer —
`cogwheel/lensing/likelihood.py::LensedRelativeBinningLikelihood._surrogate_coefficients`
— was already listed and remains.
