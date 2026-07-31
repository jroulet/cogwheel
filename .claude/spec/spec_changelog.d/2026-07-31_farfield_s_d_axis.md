---
date: 2026-07-31
bump: minor
---

### Positive-parity far-field charts use fold-adapted `(s, d)` axes

The surrogate specification now records the current far-field chart contract:
positive-parity exterior charts interpolate in gamma-resolved caustic arc
length and signed nearest-fold distance. Their required serialized arc map and
axis-schema validation reject stale coordinate artifacts at load. The former
caustic-fixed coordinates remain tile-proposal and admission coordinates only.

Macro-saddle far-field charts are intentionally not included; they fall through
to the exact engine until a per-deltoid-edge design is certified.
