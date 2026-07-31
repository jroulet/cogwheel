---
date: 2026-07-31
---

### Far-field surrogate charts use fold-adapted `(s, d)` coordinates

Positive-parity far-field chart construction, serving, serialization, and
tests now use gamma-resolved caustic arc length plus signed nearest-fold
distance. Stale or untagged coordinate artifacts refuse at load; macro-saddle
far-field requests continue to fall through to the exact engine.
