---
bump: patch
---

The module row's `ppgo_map.py` clause said `w_cert` "refuses (UNKNOWN) saddle
`rho < 1` cells" unconditionally. This build (F080) replaced that blanket
refusal with an evidence-keyed per-cell allowlist
(`CertifiedPpgoMap._saddle_rho_relaxed_floor` / `_SADDLE_RHO_RELAXED_CELLS`,
exact gamma/rho edge-equality match) and deleted the now-redundant duplicate
pre-guards in `likelihood._ppgo_cell_coords` and
`surrogate_census.characterize_sample`. Updated the clause to name the
allowlist mechanism and its sole ownership by `CertifiedPpgoMap`.

Deferred from Inspector as INS-1-003.
