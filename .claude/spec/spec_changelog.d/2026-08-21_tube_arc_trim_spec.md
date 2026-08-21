---
date: 2026-08-21
bump: patch
---

SPEC.md's BEAT-FREE TUBE RESIDUAL paragraph gained the ASTROID TUBE ARC-TRIM
sentence: `surrogate_training._trim_tube_arc` (promoted from F083) derives the
robust servable astroid sub-arc from the binding corner's live merging-pair
`Delta_tau` profile instead of charting the full cusp-to-cusp arc (whose
non-monotone `Delta_tau` makes `_merging_fold_pair` refuse near the cusps) —
low knee at `_TUBE_TRIM_DTAU_FRAC = 0.6` of the peak, both bounds stood inward
by `_TUBE_TRIM_LO_STANDOFF = 0.20` / `_TUBE_TRIM_HI_STANDOFF = 0.05` at
`_TUBE_TRIM_SCAN_POINTS = 80` resolution, parity-gated to the astroid (+1)
with saddle bands byte-identical. Closes the deferred post-commit finding
INS-1-001.
