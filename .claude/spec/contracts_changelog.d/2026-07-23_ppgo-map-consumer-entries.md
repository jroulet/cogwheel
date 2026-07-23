---
bump: minor
---

### certified_ppgo_map: register the likelihood and surrogate-training consumers

`cogwheel/lensing/likelihood.py` (`LensedRelativeBinningLikelihood._ppgo_band_split`,
which also backs `_ppgo_cell_ceiling`) and `cogwheel/lensing/surrogate_training.py`
(`train`, via the `_stratum_ppgo_boundary`/`_stratum_ppgo_ceiling` helpers) both
call `get_certified_ppgo_map` directly to read `w_trust`/`w_ceiling` per cell —
the band-split guard and the strata-trim decision, respectively. These were
consuming the artifact already; only the `ppgo_map.py::use_certified_ppgo_map`
entry point was on record. Added both as consumers so the data-flow graph
matches the code.
