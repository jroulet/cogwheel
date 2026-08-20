## 2026-08-20 (INS-2-003 caustic-feature self-falsification teeth fix)
- FIXED-FLOOR VACUITY, RE-PROBE PATTERN: a self-falsification "feature is
  load-bearing" test that asserts an absolute floor (raw_nocaustic/w_low_true
  > 1.5) can be satisfiable even when the feature is a no-op when the
  WITH-feature value already clears the floor (here 1.986 > 1.5). The correct
  teeth is a MONOTONE-WORSENING comparison between the two raw surfaces
  measured under identical conditions (both derate=1.0): raw_nocaustic >
  raw_with_caustic * 1.05. Re-probed measured values first: with-caustic
  ratio 1.986, no-caustic 2.459, so raw_nocaustic/raw_with_caustic ~1.238 —
  the 1.05 factor has real headroom (~19% below measured inflation). Teeth
  verified by patching caustic_coeff back to its ORIGINAL value (a no-op
  drop => raw_nocaustic == raw_with_caustic): the assertion trips red.
- This also lets the honest-ceiling `_measure_w_low_true` (~1.2s engine
  probe) be DROPPED from the test — the relative comparison needs no engine
  oracle, making the test purely O(1) `w_low_fit` calls. Keep the measured
  1.986 -> 2.459 ratios in the docstring as the calibrated context.
- DIFF TRAP reconfirmed in this session: the edited method sits inside a
  parallel build's uncommitted block, so `git diff` shows the WHOLE test
  class as `+` — verified the fix via targeted source-string asserts
  (old needle absent, new needle present at the expected line) plus a fresh
  pytest run, never diff isolation. ast.parse + full-file pytest (33 passed,
  3 skipped) green.
