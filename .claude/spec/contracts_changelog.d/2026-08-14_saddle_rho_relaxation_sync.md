---
bump: patch
---

`certified_ppgo_map`'s description described only the blanket F073 saddle
`rho < 1` hard-refusal. This build (F080) replaced that blanket refusal
with an evidence-keyed per-cell allowlist owned by `CertifiedPpgoMap`
(`_saddle_rho_relaxed_floor` matching `_SADDLE_RHO_RELAXED_CELLS` by exact
gamma/rho edge equality) and deleted the now-redundant duplicate pre-guards
in `likelihood._ppgo_cell_coords` and `surrogate_census.characterize_sample`.
Added a SADDLE RHO<1 RELAXATION paragraph describing the mechanism and the
single currently-allowlisted cell (gamma [1.157, 1.339] x rho [0, 0.5],
floor 19.164).

Also removed the dead consumer entry
`test_lensing_saddle_rho_guards.py::CensusBandSplitMirrorSelfFalsificationTestCase.test_site4_rho_none_is_load_bearing`
— that test class/function was deleted in the same build (its SITE-4 foil
became false once the guard it falsified was removed by design); flagged by
`sync_derived_docs.py --check` and confirmed by symbol lookup.

Deferred from Inspector as INS-1-003.
