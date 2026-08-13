---
date: 2026-08-13
section: Backlog
---

- **`_merging_fold_pair`'s CUSP-TIE GUARD WATCHES THE WRONG SIDE OF THE
  PAIR, IN SHIPPED CODE** `[→ spec]` — found 2026-08-13 while measuring the
  fold-ppGO gate; see [[FINDINGS F072]].

  `_airy_fold._merging_fold_pair` (~L319) refuses a degenerate cluster with

      tie_count = sum(1 for tau, _ in entries
                      if abs(tau - tau_high) <= _CUSP_TIE_EPS)
      if tie_count >= 2: return None

  It counts ties against `tau_high`, the SADDLE. Measured census at
  `gamma=0.5, theta=pi/2, frac=0.99` — a cusp, not a fold:

      tau=+0.15342641  morse=0  sqrt|mu|=7.089   <-- degenerate PAIR
      tau=+0.15342641  morse=0  sqrt|mu|=7.089   <-- degenerate PAIR
      tau=+0.15347645  morse=1  sqrt|mu|=9.956
      tau=+2.55878866  morse=1  sqrt|mu|=0.206

  The degeneracy is on the MINIMUM side, so the guard never fires and the
  function returns a "fold pair" built from one of the two tied minima. The
  caller then de-ppGOs one merging image and leaves the other divergent one
  in place. Measured `err_fold/err_raw` locks at 0.41 while BOTH diverge to
  1e7 as `frac -> 1`. `fold_ppgo_correction` has no cusp handoff at all,
  unlike `_uniform_arm_value`.

  ## Why this is not confined to the rung being retired

  `fold_amplification` SHIPS — `operator._uniform_arm_value` offers it before
  the exact engine. So the guard protects a live serving path, and a 3-image
  cusp merge reaching it is served by a 2-image fold form.

  ## Before fixing, PROVE the causation

  The diagnosis is inferred: the guard was read, the census dumped, and the
  constant 2.45x asymptote matched. Nobody patched the guard and re-measured.
  Do that first — patch the tie test to count ties on BOTH sides (or on the
  merging pair actually selected), re-run the same census config, and confirm
  the arm now declines and the 0.41 lock disappears. If it does not, the
  diagnosis is wrong and the real cause is elsewhere.

  ## Acceptance

  A config that currently returns a spurious fold pair, shown declining after
  the fix, with the served value before/after against the exact engine. Plus
  a check that legitimate FOLD pairs (degeneracy genuinely on the saddle
  side) still admit — the guard must not become a blanket refusal.

  ## RESOLVED 2026-08-13 (252e7c2)

  `_merging_fold_pair` (`_airy_fold.py`) now tests BOTH sides of the
  SELECTED PAIR, not `tau_high` alone: `tie_count` is the max of the tie
  count against `tau_low_best` and against `tau_high_best`. A first attempt
  that scanned every delay globally OVER-refused (gamma=1.2, r=1.0, angle=0:
  the two SADDLES tie while the selected pair is well separated and valid;
  caught by `FoldOffAxisRegressionTestCase`). Pair-scoped still catches the
  cusp, because there the tie IS the pair's own minimum. 207 passed across
  `airy_fold`, `fold_ppgo_handoff`, `ppgo_above_ceiling`, `levers`.

  CORRECTED, per FINDINGS F072: the "why this is not confined to the rung
  being retired" section above asserted `fold_amplification` SHIPS and is
  affected. That was inferred, not measured, and is WRONG — measured
  directly with the tie check disabled, `fold_amplification` DECLINES at
  both census cusp loci for reasons unrelated to this guard, so it was
  never serving a cusp merge in production. The guard's real (and correct)
  consumers are `fold_ppgo_correction` and the rung's `xi_min`.

  ## RETIRED 2026-08-13 — fragment moved to completed.d
