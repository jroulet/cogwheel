# Foreman-Lite Short-Term Observations

- INS-1-004 (cogwheel/lensing/likelihood.py ~line 1680): fixed a stale
  "byte-identical to HEAD" comment for the legacy FARFIELD_KERNEL_SUM
  serve path. Since 8h-d2 the label is demodulated by exp(+1j w t_min)
  and reconstruct_farfield re-modulates by exp(-1j w t_min), so the
  round-trip is only identical up to ~machine eps, not bit-exact.
  Reworded per Inspector's exact suggested phrasing. Single, isolated
  comment edit — no code logic touched, ast.parse confirms file still
  parses. Pattern: `search_for_pattern` on the finding's key phrase
  ("byte-identical to HEAD") found 3 hits in the file; only the one at
  the FARFIELD_DIFFRACTIVE/band_split guard (~1680) matched the finding's
  described location and content — the other two (w_trust wall ceiling,
  whole-band chart split) are genuinely still byte-identical and were
  correctly left untouched.
