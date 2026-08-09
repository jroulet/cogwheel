# Foreman-Lite Long-Term Knowledge

- A finding whose text says "Librarian-owned" / "-> Librarian:" (doc-sync,
  SPEC row) is NOT Foreman-Lite work — Foreman-Lite must not write SPEC.md.
  Decline immediately without touching files; do not re-verify the same
  no-op every pass. This finding-ID mis-route has now recurred 12x+ across
  sessions — per-pass decline is not fixing the upstream bug; escalate it
  as an orchestrator routing bug (recommend a pre-filter that strips
  "-> Librarian"-tagged findings from the Foreman-Lite queue before dispatch)
  rather than expecting future declines to resolve it.
- `rename_symbol` updates live code references but not docstring mentions
  of the old name written as `module._old_name` text — grep and fix those
  separately.
- PROBE BEFORE ASSERTING: when told to add tests for an untested module,
  measure the actual numbers first (`execute_shell_command` against the
  real functions) so every tolerance is grounded in a measured value with
  real headroom, never guessed. A measured disagreement that confirms a
  known defect becomes a concrete @unittest.expectedFailure tripwire —
  the repo convention for carrying a literal contract as an honest RED
  that flips loud when the defect is fixed.
- To isolate ONE of two independent guards, choose a fixture where the
  OTHER guard is trivially satisfied (e.g. run the parity-margin tests at
  w=0.01 so the w-scaled series guard can never trip) rather than hunting
  for a literal simultaneous boundary.
- An accuracy gate that is RED because of one unpinned constant does NOT
  invalidate invariants that are independent of it (e.g. a w->0 limit
  whose leading term carries no b1) — those stay genuinely green oracles.
  Keep the two separated in the suite; don't conflate.
- `pyflakes` is absent from the cogwheel-newlal env — fall back to
  `ast.parse` plus the actual pytest run for syntax/import verification.
- With `from __future__ import annotations` active in a module, a type
  hint can reference a not-yet-imported class as a lazy string — widening
  a `Union` type hint needs no new import; verify via `ast.parse` alone.
- When the working tree already has unrelated uncommitted changes (e.g.
  from parallel sessions), confirm your fix touched ONLY the intended lines
  via `git diff` before considering the finding resolved.
- DIFF TRAP (parallel-build variant, 2026-08-08): when your target lines sit
  INSIDE a parallel build's uncommitted block, `git diff` shows the whole
  block as `+` and cannot isolate your edit — verify via targeted source-
  string asserts instead of diff-based isolation. git stash is equally
  unusable (it also reverts the parallel build's production edits and changes
  test skip-vs-fail behavior).
- TARGETED-REVERT PROOF PATTERN (2026-08-08, INS-3-004): to prove a failure
  is independent of a specific edit when the tree has unrelated uncommitted
  changes, revert ONLY the edit in-memory (read file, remove the exact new
  lines, write, run the test, restore from a backup copy) — never git stash.
- SENTINEL THREE-CASE CONTRACT FIX (2026-08-08, `_synthetic_exterior_polar_
  chart`): a helper accepting optional (theta_to_u, u_grid) pairs must
  rewrite its sentinel block as `if both _SENTINEL: build identity-like map /
  elif either is None: force BOTH to None (raw-theta fallback) / else pass
  through` — leaking one _SENTINEL to from_values raises a confusing
  ValueError/TypeError. Note: ExteriorPolarChart stores ONLY `theta_to_u`
  (`u_grid` is consumed by from_values to build knots) — verify a raw-theta
  chart via `c.theta_to_u is None`, not a `u_grid` attribute. From_values
  validation traps for smoke fixtures: map row 0 must start at
  `theta_c_grid[0]`, `u_grid` length must equal `theta_c_grid` length, and
  n=3 violates the production >=4 nodes/axis constraint — use n=4.
- When verifying a fix that adds negative-angle entries to a min-over-angles
  ceiling: confirm each added angle's w_star exceeds the wall constant to
  prove the minimum is unaffected; if _w_star uses raw (non-abs) angle,
  negative angles in the lossy regime are non-restricting by design.
- For stale-comment fixes (e.g. a literal count in a comment after a
  constant was reduced), a single `replace_content` call is sufficient —
  no structural changes needed; verify via read-only check before and after.
- TWO-GATE CORRELATION near cusps: near-axis/near-cusp configs have Im(tau_c)
  → 0 AND image separation → 0 simultaneously — the two guards are physically
  correlated. Isolating `_GHOST_SEPARATION_MIN` reachability tests requires
  patching BOTH `_GHOST_DECAY_IM_THRESHOLD` AND `_GHOST_SEPARATION_MIN` to
  zero; patching only the separation constant leaves the decay gate blocking.
- ADMIT CONFIG CRITERION: a valid ghost-ADMIT fixture must satisfy all three:
  (1) decay gate passes, (2) separation gate passes, AND (3) ghost subtraction
  actually helps (resid(MINUS_GHOST) <= resid(KERNEL_SUM) — the DoNothing
  check). A config passing both gates can still have counterproductive ghost
  subtraction — verify criterion 3 empirically when selecting ADMIT fixtures.
- XDIST TREE-GATE CRASH (recurring infra, NOT code): pytest-xdist workers
  can crash with an `AssertionError: worker_workerfinished` during parallel
  test collection/distribution. This is a known xdist race condition when
  workers die during tree-gating. Symptoms: random test abort mid-suite,
  non-reproducible, usually on larger test files. Mitigation: re-run the
  failing suite without -n (serial); if it passes serial, it's this bug.
  Not a code defect — do not chase.
- SIDECAR CALLBACK SILENT DEATH (>1hr builds): the sidecar callback (used
  for build progress/heartbeat monitoring) dies silently on builds exceeding
  ~1 hour. Cause unknown — possibly a timeout/keepalive expiry in the
  callback transport. Symptoms: build continues to completion but no
  progress updates after the ~1hr mark; final result IS returned. Not a
  correctness issue (build output unaffected) but a monitoring gap. No fix
  landed; workaround: check build logs directly for long builds rather than
  relying on callback heartbeats.
- `stable_gamma_bands` return type: returns `(stable, dropped)` where each
  entry in `dropped` is a `(lo, hi)` float tuple; sum dropped widths with a
  generator expression `sum(hi - lo for lo, hi in dropped)`.
- CARRY-FORWARD DOC-FINDING GREP LESSON (2026-08-08, INS-4-001): a carried
  doc-staleness finding (docstring said "Used by the wedge-interior chart"
  though `_validate_theta_to_u` ALSO serves LobeInteriorChart) was missed by
  earlier greps because the needle was a PARAPHRASE without the word "lobe".
  When a doc finding is carried across reviews, grep the EXACT docstring
  sentence from the finding, never a paraphrase — and confirm BOTH callers
  by search before editing.
