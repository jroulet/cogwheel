---
section: Backlog
---

- **Never pin a symptom with a reachable-red; pin the invariant**
  `[housekeeping]` — process rule drawn from the F022 post-mortem, to be
  folded into `.claude/crew/test_dev.md` and `.claude/crew/inspector.md`.

  `test_unpatched_positive_box_build_raises_carrier_discontinuity` existed to
  assert that a coarse box TRIPS the far-field carrier guard. Its stated intent
  was honourable — prove the fixtures' bypass was not masking a guard that had
  silently stopped firing. Its effect was the opposite: it promoted a false
  positive to a specification. Once written, fixing the guard REGISTERS AS A
  REGRESSION, and the only reason the defect was found is that this test failed
  when the guard was corrected.

  Rule: a reachable-red belongs on an INVARIANT ("a genuinely discontinuous
  tile is rejected", certified synthetically in
  `FarfieldCarrierContinuityGuardTestCase`), never on a SYMPTOM ("this specific
  real fixture is rejected"). If liveness of a guard needs proving, prove it
  against a constructed pathological input, not against a production input
  whose rejection is the thing under suspicion.

  Second rule, same post-mortem: RECURRING ACCOMMODATION IS A SIGNAL ABOUT THE
  THING BEING ACCOMMODATED. Four independent bypasses accumulated for this one
  guard (three `_skip_carrier_guard=True` sites plus a
  `_from_engine_without_carrier_guard` mock-patch helper), each with a written,
  locally-reasonable justification citing real accuracy evidence. No mechanism
  existed to ask whether four accommodations meant the guard was wrong, because
  each build only sees its own. Proposed check for the Inspector: when adding a
  bypass/skip/xfail for a named production guard, grep for existing bypasses of
  the SAME symbol; the second one escalates to a review of the guard rather
  than a third bypass.

  Third rule, added 2026-07-28: A TEST WHOSE PREMISE IS "HEAD IS THE
  PRE-CHANGE REVISION" CANNOT BE BOTH GREEN AND MEANINGFUL IN THE TREE IT IS
  COMMITTED TO. `test_lensing_farfield_envelope.py` carried a `_head_module`
  helper that imported a module via `git show HEAD:<path>` and compared it
  against the working tree, to certify the 8h-d2 far-field changes were
  additive. Before the commit it passed and meant something; after the commit
  HEAD *is* the branch, so `TubeByteIdentityTestCase` compared
  `surrogate.py` to itself and passed unconditionally, while
  `FarfieldTelescopingRoundTripTestCase` errored outright
  (`reconstruct_farfield() missing 1 required positional argument: 't_min'`).
  It had no window in which it was correct in its own repository.

  Worse, it is invisible to the pre-commit gate BY CONSTRUCTION: the full
  suite was green immediately before the commit, because HEAD was still the
  old revision then. The breakage appears only in the NEXT run. Retired the
  apparatus rather than re-pinning it to a fixed SHA, which defers the rot
  instead of removing it. Migration equivalence belongs in a throwaway probe
  the driver runs while the migration is in flight, never in a committed
  test. What belongs in the suite is the intrinsic claim -- here, that
  `reconstruct_farfield` reproduces the engine's exact total.

  Fourth: the build that added the fourth bypass had ALREADY MEASURED the
  refutation — its docstring records `n_gamma in {6, 8, 12, 16}` all raising at
  `~3.1 rad` — and filed it as an unavoidable "integration tension". A step
  that does not shrink under refinement is not an under-resolution problem, by
  definition. The refinement sweep is the discriminator (carrier shrinks like
  `1/n`, null pins at `pi`) and should be run before any conclusion about a
  phase-based guard.
