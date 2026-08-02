---
bump: patch
---
### Born-residual wiring infrastructure landed

SPEC.md § Born carrier: replaced "STILL NOT wired" status sentence with
current reality — the fact-4 slot in `likelihood._surrogate_coefficients`
is now wired. `BornResidualChart` (frozen 3-D interpolation dataclass,
`cogwheel/lensing/born_residual_chart.py`) plugs into the likelihood object;
when attached, the slot reconstructs `F_carrier + R(w; gamma, rho)`. When
`None` (default), annulus draws fall through to the exact engine as before.
The trained chart itself remains a TRAIN_TIER artifact.
