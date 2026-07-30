---
date: 2026-07-30
section: Testing
---

### Every `git show HEAD` test oracle retired (F043 / F045)

No test in `cogwheel/tests/` reconstructs code or constants from a previous
commit any more. Fourteen HEAD-dependent tests across four files were audited
and removed or re-pointed; the pre-commit guard
(`.claude/hooks/check_head_relative_tests.py`) now also catches a CALL to a
HEAD-oracle helper, not just a freshly written `git show HEAD:` line.

**The audit that justified deletion.** On the sweep date `git show HEAD` and
the worktree were BYTE-IDENTICAL for `geometry.py`, `likelihood.py` and
`surrogate_training.py`. Every "byte-identity against HEAD" test was therefore
comparing a module against an exact copy of itself, and
`test_served_bar_is_a_tightening_not_a_widening` was asserting
`0.001 <= 0.001` while its own docstring claimed it checked against `3e-3`.
Fifteen tests were passing and none of them could fail.

**Removed** (all vacuous by the above):
- `test_lensing_ghost.py`: `RealImageByteIdentityTestCase` (image positions,
  delay/magnification/Morse, image kernel, Morse census) and
  `test_byte_identity_gate_catches_a_one_ulp_perturbation`, whose reachable-red
  step asserted `abs(nextafter(x, inf) - x) > 0` -- a fact about floats,
  vacuous in principle rather than merely on the day.
- `test_lensing_levers.py`: the Lever 1 and Lever 2 value-preservation and
  self-falsification classes. This module's premise was "HEAD is the commit
  these levers sit on top of, and therefore the trusted oracle"; that expired
  when Build 8f committed.
- `test_lensing_caustic_cusps.py`: `CuspWindowByteIdentityTestCase` and
  `test_spec2_window_table`, both already `@unittest.skip` shells for F043.

**Kept and re-pointed** (the claim was real, only the oracle was wrong):
- `test_wrong_image_count_is_detected` now builds its mutant from the worktree
  finder. Dropping an image needs no cross-version oracle, so it is stronger
  than before, not weaker.
- The far-field bar comparison now reads a frozen `PRIOR_FARFIELD_EPS_MAX =
  3e-3`. A historical value is a constant, not a query.

**Untouched, and verified untouched:** `test_lensing_ghost.py`'s protected
Richardson finite-difference `det H_c` oracle and its AST independence guard
never used `_head_geometry`; Lever 3's node-parallel comparison is the model to
copy, since its oracle is the same grid function with the njit map swapped for
its `.py_func` -- an independence that lives entirely in the worktree.

**Why this was a cleanup and not a build:** a build would have introduced new
tests in the very construct being retired, and the deletion criterion was an
audit result, not a design decision.
