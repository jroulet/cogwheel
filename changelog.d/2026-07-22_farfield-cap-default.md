---
date: 2026-07-22
---
### Far-field tile cap defaults to uncapped

`TrainingConfig.max_farfield_regions` now defaults to ``None`` (no cap):
the tiling itself bounds the chart count, and a default-config run no
longer silently truncates coverage to a single tile (accepted Build 8g
inspector finding).
