---
section: Backlog
---

- **Dropped metamorphosis gamma slivers get NO chart of any kind, and the code
  comment says otherwise** `[→ spec]` — a first-class census fall-through
  bucket with no owner. Found by backlog audit 2026-07-28; it had never been
  written down.

  `surrogate_training.py` calls `stable_gamma_bands(..., min_width=
  config.min_gamma_band)` (default `0.02`) and DISCARDS every topology-stable
  gamma sub-band narrower than that. The training loop then calls
  `_train_band_charts` only over the SURVIVING sub-bands — and
  `_train_band_charts` is what builds BOTH the tube charts AND the
  far-field/interior tiles for a band.

  So a dropped sliver receives no chart at all: the entire source plane and the
  entire `w` band at those gammas fall through to the exact engine. Not a
  degraded serve — no serve.

  **The in-code comment asserts the opposite and is WRONG** (near
  `surrogate_training.py:2982`):

      "metamorphosis slivers are dropped (they fall through to
       far-field/exact serving)"

  There is no far-field to fall through to, because the far-field tiles are
  built inside the same per-sub-band call that was skipped. Fix the comment in
  the same change that fixes the behaviour — a comment that misdescribes a
  known gap is how the gap stayed invisible.

  `surrogate_census._FALLTHROUGH_CATEGORIES` already lists `'dropped-sliver'`
  alongside `'cusp-window'` and `'refusal-ball'`, so the census would REPORT
  this bucket — but no fragment tracked closing it, and its magnitude has
  never been measured.

  Owed, in order:
  1. MEASURE it first. Each sliver is at most `min_gamma_band = 0.02` wide, but
     the COUNT is data-dependent (it is whatever the astroid/deltoid
     metamorphosis structure produces across `gamma` in (0, 1.6) on both
     parities) and has never been counted. Total prior mass in dropped slivers
     is the number that decides whether this is a rounding error or a real
     hole. Cheap: run the band splitter across the prior and sum the dropped
     widths.
  2. Fix the comment regardless of (1).
  3. If the mass is non-negligible, decide the treatment: widen the bands to
     absorb the sliver (accepting topology mixing within a chart), serve the
     sliver from a neighbouring band's charts with an accuracy check, or make
     it an explicit named refusal so it is counted rather than silently
     exact-served.

  Do NOT close this by lowering `min_gamma_band`: the threshold exists because
  a topology-unstable band cannot be tiled coherently. The question is what
  serves the sliver, not how to make the sliver disappear.
