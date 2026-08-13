---
section: Backlog
---

- **REMOVE `FARFIELD_KERNEL_SUM_MINUS_GHOST` — it has no producer and its
  survivors actively mislead** `[→ spec]` — owner-directed 2026-08-13. Ride
  this with the deferred tier-2 build
  ([[lensing_saddle_envelope_is_negligible_at_high_w]]), which is already
  opening the same files; do NOT spend a build on it alone.

  Full diagnosis in FINDINGS F065. In short: the label's ONLY producer was the
  macro-saddle origin-polar exterior tiler (`force_minus_ghost =
  _exclude_near_cusp(...)` under `if parity == -1`), retired by `4c7dc92`
  when that tiler was replaced with the lobe-local exterior. Positive parity
  never stamped it — `surrogate_training.py` ~L5073 hardcodes
  `force_minus_ghost = False` with a comment saying so.

  ## What to remove

  - `FARFIELD_KERNEL_SUM_MINUS_GHOST` from `channels.py`: the constant, the
    `__all__` export, and its membership in `KNOWN_FARFIELD_DEFINITIONS` and
    the other definition sets.
  - The serve-side branch at `likelihood.py:1871`
    (`if definition == FARFIELD_KERNEL_SUM_MINUS_GHOST:`) — currently
    unreachable.
  - The `force_minus_ghost` parameter from `_build_farfield_chart` and every
    tile-dict key that threads it (`surrogate_training.py` ~L3024, 3037, 3096,
    4391, 4517, 5073-5079, 5710-5745).
  - `test_lensing_ppgo_midw_and_minus_ghost.py`'s MINUS_GHOST classes. NOTE
    the file ALSO covers the astroid mid-`w` ppGO band, which STAYS — split
    the file or delete only the ghost classes, and do not lose the ppGO
    coverage. That file also contains the cross-class fixture borrow fixed on
    2026-08-12; keep that fix.

  ## What must NOT be removed

  `ghost_kernel`, `farfield_ghost_term` and the ghost machinery in
  `geometry.py` are LIVE — they serve the two-real-image ppGO + ghost branch
  above the Born band split (`w * Delta_tau >= RHO_END`). Only the far-field
  chart LABEL and its dead plumbing go.

  ## Load-bearing check before deleting

  A stored artifact carrying the label would fail to load once it leaves
  `KNOWN_FARFIELD_DEFINITIONS`. Confirm no shipped `.npz` stamps it —
  `cogwheel/data/*.npz` plus any local training output — and say so in the
  build report. Expected to be none, since the producer is gone, but a
  hard-refusing loader turns a wrong expectation into a broken load.

  ## Acceptance

  `git grep FARFIELD_KERNEL_SUM_MINUS_GHOST` returns nothing outside
  CHANGELOG/FINDINGS history; the astroid mid-`w` ppGO tests still pass; the
  ghost branch above the band split is untouched and its tests still pass.
