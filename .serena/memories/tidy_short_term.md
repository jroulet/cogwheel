# Tidy Short-Term Observations

- 2026-08-13 lensing pass (surrogate_training.py, surrogate.py,
  chang_refsdal/channels.py): `tidy_mechanical.py --check`'s printed line list
  is CAPPED AT 6 (`longs[:6]` in `_long_lines`/`tidy_file`) even though the
  count in the summary is the true total. For files with >6 long lines, get
  the full list yourself (e.g. a one-line python len()-scan), don't trust the
  printed sample as exhaustive.
- Long-line judgment split that worked well: wrap CODE lines (any width over
  79, no exceptions) but leave PROSE (comment/docstring) lines alone unless
  >=85 cols or a trivial 2-line word-shift resolves it cleanly — many
  surrogate_training.py overages sit in one ~1000-line prose-heavy function
  (`_train_band_charts`) where paragraphs are already hand-wrapped at ~78-80
  cols; chasing every 80-84 line there would mean touching dozens of lines
  for near-zero readability gain. Flagged the function's size itself as the
  real Q4 finding instead of hand-wrapping around it.
- Implicit-string-concatenation lines (adjacent string/f-string literals
  split across lines, e.g. a multi-line ValueError message) were left alone
  even when >82 cols: rebalancing the split point preserves the joined
  value but changes each individual literal token's content, which reads as
  a violation of the "never change string literal content" hard constraint.
  Two instances found and left: surrogate_training.py ~5391 (post-edit line
  number) and chang_refsdal/channels.py:1505.
- Public-API-before-private-helper ordering is violated in all three lensing
  files, but it reads as deliberate bottom-up structuring (private
  primitives → public functions → public classes), not disorder — e.g.
  channels.py builds `_physical_kernels`/`_channel_switch` before exposing
  `farfield_envelope_from_partition`/`born_carrier_from_partition`; the same
  pattern repeats in surrogate.py and surrogate_training.py. Reported, not
  reordered, per the "diff risk outweighs benefit" rule for large modules.
