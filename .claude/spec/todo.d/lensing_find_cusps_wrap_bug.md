---
section: Backlog
---

- **NEXT-SESSION ORDER 3/7 — FIX `_find_cusps` WRAP ARITHMETIC — BLOCKS
  THE TRAINING CAMPAIGN**
  `[→ spec]` — F079: periodic index wrap + linear angle arithmetic gives
  the theta = 0 cusp a 1.5-pi dip span, `_make_arc` returns None for both
  adjacent arcs, and half the astroid fold ring gets no tube chart,
  silently. Fix the wrap (angle differences mod 2 pi), add an
  arcs-survive-the-tiler pin (4 cusps -> 4 arcs on the astroid; the
  topology check must count ARCS, not just cusps), and re-measure the
  theta = 0 window against a window-interior baseline (~0.11-0.13 rad at
  the tested gammas). In the same change: retire `_CUSP_ARM_COVERAGE`,
  `_SADDLE_CUSP_ARM_COVERAGE`, the three measure_* scripts, and the
  `_tube_serves` shrink (measured inert, wrong units — F079 body); update
  the census `cusp-window` category note; test debt enumerated in the
  measurement report (test_lensing_surrogate ~5127-5295, surrogate_training
  D2a/D2b ~6001-6260, cusp_arm_coverage suite, census_dry_run.py:28,
  calibrate_ppgo_rung.py:48,160,222). MUST land before
  train_lens_surrogate.py runs, else the artifact ships with the
  half-ring hole.
