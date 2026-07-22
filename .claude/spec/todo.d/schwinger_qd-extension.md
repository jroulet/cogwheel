---
section: Backlog
---
- **Quad-double Schwinger extension (owner-approved 2026-07-22)** `[→ spec]` —
  extend the Schwinger t-integral core from double-double to quad-double
  precision, moving the certified w ceiling from ~60 to ~155 (measured
  arithmetic wall: 0.341 digits/unit-w; dd carries 31.9 digits, qd ~63).
  Intercepts 73% of the prior's hard nodes (median hard w = 105, config-level
  census 2026-07-21). Certification pattern unchanged: certified-or-refuse,
  N-vs-2N node doubling, refusal vocabulary untouched. numba compatibility
  of the qd arithmetic is the central risk; keep the dd path as the w <= 60
  fast path. Enables saddle-parity chart labels (and far-field tiling
  coverage) up to m_lens ~ 1200 Msun (vs ~460 today).
