---
date: 2026-07-28
section: Backlog
---

- **Frame-invariant label round-trip costs ~3e-11 on near-fold configs** —
  RESOLVED by fixing the BOUND's currency, not the tolerance's value. The
  `MorseSignMaskTestCase` telescoping xfail is retired and the test passes on
  its merits.

  The telescoping identity rebuilds `F` by adding real carriers back onto the
  far-field remainder, so working precision is set by the largest
  intermediate, `|E_tilde|`, while the answer is only `|F|`. Achievable
  accuracy is therefore `eps * max|E_tilde| / max|F|` — a condition number,
  not a constant. Next to a fold the near-degenerate image's kernel diverges,
  `|E_tilde|` reaches 2.55e5 against `|F| = 2.78`, and the floor sits at
  2.04e-11 — ABOVE the flat 1e-11 bound the test asserted. The measured error,
  1.66e-11, was already BELOW that floor: the reconstruction was running at
  the limit of double precision and the bound was simply asking for more than
  arithmetic can deliver.

  Measured across 11 configurations (deep interior, mid, near-fold,
  across-caustic, exterior, at gamma 0.3/0.5/0.7): realized error is 0.11x to
  1.53x the floor — always at or below it, never worse. `_telescoping_floor`
  now computes the floor and both call sites assert against
  `max(flat_bound, 4 * floor)`, so the flat bound still governs every
  well-conditioned fixture (the interior case keeps its 1e-12 and passes at
  1.6e-16) and only the ill-conditioned one relaxes, to the limit physics
  allows.

  This is STRONGER than the constant it replaces: it asserts that double
  precision could not have done better, on every fixture, rather than picking
  a number that fits the easy ones. Falsification checked by mutation — with
  the bound at 8.16e-11 the test still fails on a `t_min` perturbation of one
  part in 1e6 (error 6.3e-2, nine orders over the bound) and on `t_min = 0`
  (6.2e4). It remains exquisitely sensitive to the frame bugs it exists to
  catch.

  Not done, and deliberately: the underlying conditioning is real. A
  demodulated label that grows to 1e5 near a fold is poorly conditioned
  exactly where the physics is hardest. Reconstructing in the min-relative
  frame on that path measures 4.9e-12 and would avoid the round-trip multiply
  entirely, but that is a serve/label data-flow change and should be decided
  on its own merits rather than smuggled in as a test fix.
