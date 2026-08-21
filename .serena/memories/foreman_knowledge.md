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
- STALE-VALUE GREP + EXACT-STRING VERIFY (2026-08-09, INS-6-002): after a
  constant's value change (e.g. `_R_PPGO_ERROR_CONST` 2.0->50.0 moving
  r_ppgo_min 54.3->464.16), grep the WHOLE file for the old VALUE STRING
  ('54.3'), not the finding's stated line range — it missed 2 of 4 stale
  sites (a comment + a test docstring). PROBE BEFORE ASSERTING the claimed
  threshold (finding said ppGO fires at w>=5000 but measured control-radius
  crossing is w~3980; use the measured value in comments). Verify a
  comment-only fix via EXACT-STRING asserts (old value absent, new value
  present) when git diff shows the whole block as '+' (DIFF TRAP,
  parallel-build variant) — diff-based isolation is unusable there.
  ast.parse is the syntax check when pyflakes is absent.
- PARTIAL-PATTERN VERIFY TRAP (2026-08-11, INS-17-001): a guard assertion
  for "old claim gone" must match the EXACT old string from the finding — a
  partial pattern like `< 1e-12` still matches sibling lines (e.g. the
  PM_CERT_RTOL = 1e-12 constants at line 128). Use the full span
  (`|F_DD - F_mpmath| / |F_mpmath| < 1e-12`) in the assertion.
- DEAD-ASSIGNMENT CASCADE (2026-08-11, INS-18-001): after deleting an
  algebraically-unreachable branch (a ternary whose condition is forced by
  an enclosing guard), grep the WHOLE file for the variable it fed — a
  leftover dead assignment (`tol_float = float(_CERTIFICATION_TOL)`) is the
  same defect family as the deleted branch.
- OMIT `cwd` IN `execute_shell_command` (2026-08-13): it already defaults to
  the worktree the session runs in, which can differ from the project path
  quoted in a task's file paths; passing an absolute cwd guess fails with
  FileNotFoundError. Use relative paths matching `find_file`'s output.
- WHEN SERENA IS DOWN (it has died twice under memory pressure / regex
  backtracking), the working stack is: native Read/Edit/Write for `.claude/`
  paths (hook-exempt), `conda run -n <env> python <script_file>` for
  everything else (top-level Bash allow-list), `git show HEAD:<path>` to
  read a gated source file, and `git mv` since `mv`/`rm` are not allowed.
  Never a heredoc — heredoc stdin can execute as empty with rc 0. Full
  procedure in `mem:librarian_knowledge`.
- SHELL QUOTING TRAP (recurring): inline `python -c "..."` verification with
  backticks/`<` in assertion strings is mangled by bash double-quote
  processing and can produce a misleading AssertionError at a line that is a
  print statement; use a heredoc'd temp script (`cat > /tmp/... << 'PYEOF'`)
  for any inline-python verification containing backtick/`<` characters.
- TOP-LEVEL BASH SANDBOX BLOCKS `cd`-CHAINED COMMANDS AND `grep` OUTRIGHT
  (2026-08-14, error text: "USE SERENA for shell commands"): invoke git/ls/
  stat/etc. directly with `-C <path>` (never a leading `cd`), and route all
  content search through `mcp__serena__search_for_pattern` instead of grep.
- SWEEP SIBLING DOCSTRINGS AFTER A GUARD-REMOVAL FIX (2026-08-14): when a
  finding names one stale docstring caused by a removed guard, grep the
  whole file for the same stale phrasing — sibling tests/comments describing
  the same removed mechanism are the same defect, not a separate finding.
  Verify any numeric claim in the corrected prose (w_trust, thresholds) with
  a fresh pytest run before trusting it, not just internal consistency.
- DEAD-FIELD REMOVAL VERIFICATION RECIPE (2026-08-17, serve_route_census):
  before deleting an unused dataclass field + its assignment,
  `search_for_pattern` the field name file-wide to confirm the two sites are
  the ONLY occurrences; after removal, re-run the same search (expect zero)
  plus `ast.parse` + a live import + a `dataclasses.fields()` count check as
  the mechanical proof the removal was both clean and syntactically valid.

## 2026-08-18 (INS-1-003, born_carrier_omitted_term ZeroDivisionError)
- SENTINEL-GUARD EXCEPTION COVERAGE: a helper returning a documented
  sentinel (e.g. math.inf) for one domain-boundary exception (ValueError at
  the true origin) can still raise an UNCAUGHT different exception
  (ZeroDivisionError, e.g. `1.0/math.sqrt(abs(det_a))` at det_a==0 on the
  caustic boundary) for a different degeneracy -- audit every raise path in
  the wrapped helper, not just the one the original guard anticipated, and
  broaden the except clause (or add an explicit pre-check) to cover it.

## 2026-08-20 (INS-4-001/002, coverage-claim qualifications)

- COVERAGE-CLAIM × FENCE-INTERACTION AUDIT: a coverage claim ('throughout')
  in a grid docstring must survive a fence audit — when rows of an interior
  cell are FENCED OUT in some directions (cusp-direction thetas at r=0.2:
  16/32 rows for (0.2,0.2), 8/32 for (0.3,0.2)), 'throughout' inverts into
  exactly the coverage gap the tests name. Qualify by DIRECTION (smooth
  off-cusp directions vs near-cusp thetas fenced out), never by averaging.
  Same class: a de-rate docstring 'never over-serves' must be qualified to
  the calibration grid + held-out probes ('extrapolated off-grid points
  can over-serve').

## 2026-08-20 (INS-1-002, low_w_diffractive_chart docstring)

- DOC-FIX CANONICAL-SOURCE RULE: when a chart's module/evaluate docstrings are
  stale but the likelihood serve docstring (and the training script) describe
  the representation correctly, take the formula VERBATIM from that canonical
  production source (`r_pure = f_pure / (sqrt(mu_pure) * prefactor_c(w))` and
  `F_serve = mass_sheet_phase * prefactor_c(w) * sqrt_mu_full * r_pure`) —
  never re-derive a paraphrase. Here the stale "anchor" framing
  (`F_serve = r * anchor`, anchor = single factor sqrt(mu_macro)*exp(phase))
  was subtly wrong because the real normalization is a PRODUCT of two analytic
  factors (sqrt(mu_pure) AND prefactor_c(w) = C(w), |C| ~1.4 at w=0.5); the
  fix names both. Same-class trap as the "paraphrase grep miss" entry: grep
  the EXACT string from the finding, not a paraphrase.
- Doc-only fix verify recipe: ast.parse (pyflakes absent in cogwheel-newlal)
  + live import + read-back of both edited regions; no pytest run needed —
  zero code behavior changed.

## 2026-08-21 (INS-2-001/002, dead-field + dead-accessor removal in _pearcey_cusp.py)

- DEAD-ACCESSOR REMOVAL VARIANT of the dead-field recipe: an unused public scalar that is an exact duplicate of an existing grid-form entry (cusp_uniform_reference([w])[0]) can be DROPPED (Inspector-sanctioned) instead of pin-tested — verify via dataclasses.fields() list + hasattr-absent + __all__ membership + ast.parse + live import (diff isolation is unusable in a file carrying parallel uncommitted work: git diff shows the whole refactor block as `+`).
- When removing a dataclass STORAGE field, keep the LOCALLY-COMPUTED values that feed derived fields (branch/vertex/phi_ssr feed vertex/curvature/c4) — only the storage is dead; removing the local computations breaks the derived fields.
- Mid-file function deletion via regex `\ndef NAME\(.*?<last-body-line>\n` with an EMPTY repl leaves the correct two-blank-line spacing; a single-newline repl leaves THREE blank lines (needs a cleanup pass).
- SPEC-staleness scan after dropping a symbol: grep the SYMBOL name — SPEC goes stale only if the QUANTITY itself disappears; a quantity still computed locally (phi_ssr in the F074 serving-ladder formula) leaves no staleness.


## 2026-08-21 (INS-3-002, dead-import removal)

- DEAD-IMPORT REMOVAL RECIPE (single-line): `search_for_pattern` the
  symbol across the WHOLE file to confirm the import line is the ONLY
  occurrence (sibling files importing the symbol via their OWN imports are
  untouched); one replace_content; verify via ast.parse + live import +
  hasattr absent/present asserts; no pytest for an import-only zero-behavior
  change. SPEC-staleness: grep the symbol name — stale only if the
  QUANTITY disappears, not if a dead import is dropped.
- TOOLING GOTCHA: a redundant second replace_content with the SAME needle
  raises ValueError (no matches) after the first already applied — don't
  read that as an edit failure; re-read the file to confirm the first edit
  landed.
