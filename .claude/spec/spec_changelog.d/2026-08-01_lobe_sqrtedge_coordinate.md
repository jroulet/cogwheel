---
date: 2026-08-01
bump: patch
---

Lobe-interior charts gain optional `theta_to_s` axis map for wedge-edge
reparametrization (`s = sqrt(span) - sqrt(theta_max - theta)`). New
`_LOBE_AXIS_SCHEMA` tag and `_LOBE_ARC_MAP_SIZE` constant. V1 charts
(theta_to_s=None) retain byte-identical behavior.
