# Librarian Short-Term Observations

## 2026-08-21 (low_w_diffractive fold/cusp reference doc sync, INS-1-004/2-003/3-001)

- RESIDUAL-REPRESENTATION STALENESS SPANS TWO BUILDS, NOT ONE: the low-w
  diffractive chart's residual anchor changed TWICE with zero doc edits —
  (a) previous build 7acffea replaced the point-mass `prefactor_c(w) = C(w)`
  anchor with the Airy fold reference AND re-gridded the frequency axis from
  `log w` to `w**(2/3)` / schema `v1`->`v2`, then (b) THIS build added the
  Pearcey cusp fallback (`airy_fold_reference` -> `fold_cusp_reference` +
  `_NON_VANISHING_MIN_RATIO` guard). Inspector flagged only the F_ref bit;
  the axis/schema bits were stale in the SAME sentence from the earlier
  commit. When a finding names one stale clause in a long description,
  audit the WHOLE sentence (and the artifact's schema/fields) against the
  code, not just the flagged clause.
- `cusp_amplification`'s ppGO fast-rung ordering ("returns before any table
  or quadrature lookup", SPEC.md line 54) was TRANSIENTLY broken by the
  WP1 bundle extraction then RESTORED in-build via `_cusp_controls` +
  deferred `_cusp_uniform_at_w` — the working-tree code is the source of
  truth; the coder change reports recorded both the breakage and the fix.
  Cross-checked the actual `cusp_amplification` body (ppGO rung early-return
  before `_cusp_uniform_at_w`) — SPEC claim stays TRUE. Also the residual
  served is `r_new = f_pure * sqrt(1-gamma'^2) / F_ref` (macro normalization
  STAYS; only the anchor is replaced) and `covers()` checks the w band inside
  the trained `w**(2/3)` range — both now reflected in SPEC.md + DATA_CONTRACTS.
- SPEC.md `\|` escapes: verified the edited row byte-level (single backslash
  + pipe) — consistent with the existing 32 `\|` escapes. There are 2
  PRE-EXISTING unescaped pipes in the same giant row (`|F_ref|` in the
  BEAT-FREE TUBE RESIDUAL sentence, `max|F|` in the F070 paragraph) that
  break table rendering — pre-existing, unrelated to this build; left as-is,
  flagged for a future housekeeping pass.
- POST-COMMIT BACKLOG ABSORBED INTO IN-DAG RUN: `.claude/sync_issues.json`
  exists (accumulated 08-18..08-21, ~25 pending commits) but this run is the
  BUILD-pipeline librarian (no commit). Its 2 deferred doc findings:
  INS-4-001 (retired `w_low` truncation formula) already fixed by the
  08-20 sync — verify-only; INS-1-001 (`_trim_tube_arc` astroid arc-trim not
  in SPEC) was genuinely open — FIXED here (SPEC sentence + spec_changelog
  fragment 2026-08-21_tube_arc_trim_spec, spec 0.49.3->0.49.4). The
  sync_issues.json itself was NOT deleted (post-commit hook still owns it);
  next post-commit run should find INS-1-001 resolved. Do NOT delete
  sync_issues.json in build mode.
- Spec 0.49.2 -> 0.49.4 (patch x2: low_w fold/cusp reference + tube arc trim),
  schema 3.4.0 -> 3.4.1 (patch). New changelog fragment
  changelog.d/2026-08-21_low_w_diffractive_fold_cusp_reference.md.
- NEW FRAGILE CROSS-REF: SPEC + DATA_CONTRACTS now cite `fold_cusp_reference`
  and its `_NON_VANISHING_MIN_RATIO` guard. If the Airy-primary/Pearcey-
  fallback pairing or the guard is ever dropped, BOTH surfaces go stale
  together (same family as the `_PPGO_BAR_DIVISOR` cluster). Also
  `cusp_uniform_reference_grid` (new public in `_pearcey_cusp.__all__`,
  cluster-only uniform form, live quadrature, no serving gates) is the
  Pearcey leg's engine — SPEC's cusp paragraph does NOT name it; add for
  discoverability only if the cusp paragraph is next touched.
- INSPECTOR TRIVIAL FLAG NOT FIXABLE BY ME (code docstring, read-only):
  `fold_cusp_reference`'s guard-rationale docstring names only the P~0
  decline mechanism (interior cusp cells), missing the far-exterior decline
  the self-falsification test actually exercises (cluster fully resolves
  above w~7 -> `matched_delays` empties -> `cluster_sum -> 0` ->
  `min|F_ref| == 0`). A reader could conclude exterior cusp cells are never
  guard-declined. Code-side fix; next code-touching build should extend the
  docstring. Flagged, not edited (hard rule: no code edits).
