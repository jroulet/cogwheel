---
date: 2026-08-07
---

### Lensing surrogate training: optional m_lens_range mass-stratum restriction

`PriorBox.from_prior_classes()` and `train()` in
`cogwheel/lensing/surrogate_training.py` now accept an optional
`m_lens_range` parameter (`(m_lo, m_hi)` in Msun). When given, the
lens-mass prior box is restricted to that mass/w stratum instead of the
full prior mass range, so a per-region probe can train a single mass/w
stratum through the production training path rather than reimplementing
it. `None` (the default) keeps the full prior mass range and is
byte-identical to the previous behavior.
