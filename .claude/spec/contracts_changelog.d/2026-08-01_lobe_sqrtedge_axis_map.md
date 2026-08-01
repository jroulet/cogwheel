---
date: 2026-08-01
bump: patch
---

`lens_amplification_surrogate` artifact: lobe-interior charts now persist a
`theta_to_s` axis map (2 × N_map array) under the new
`_LOBE_AXIS_SCHEMA = 'lobe_local_offset_rholobe_thetalocal_sqrtedge_framewinv'`
tag. Legacy V1 charts (no map) load under the old schema tag with
`theta_to_s=None`. The map is gamma-independent (depends only on tile bounds).
