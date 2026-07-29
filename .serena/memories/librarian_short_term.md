## 2026-07-29 post-commit sync — retired operator series / eta gate (commits 16f7ec0..4318dab)

Scope: 5 of 6 queued commits in .claude/sync_issues.json were test-only or
FINDINGS-only (16f7ec0, c7d2cff, 27f5cda skip per rule; 53d3d36 was already
committed by a prior session). The two with real cogwheel/ production diffs:
c1a552f (shear-free gamma'==0 exit moved off the legacy series onto a closed
form) and 4318dab (eta/distance-to-caustic leg added to `select_branch`;
`CancellationError` class deleted entirely; routing-pin consolidation).

STALENESS FOUND AND FIXED (SPEC.md, .claude/spec/SPEC.md, 8 targeted edits in
the single giant "Microlensing engine" table row):
1. Limitations bullet still described "wave-branch contraction... L in
   [~30,48] certified-or-refused... CancellationError" as CURRENT — that
   framing predates even Build 8d and describes a mechanism (the legacy
   dd/1F1 series as a live wave evaluator) that no longer exists at all.
   Rewrote to name the actual current ceiling/refusal (Schwinger quadrature,
   `w<=60`, `SchwingerCertificationError`) and added the eta leg.
2. HOMOGENIZATION (Build 8d) paragraph said the legacy series was "DEMOTED to
   the shear-free exit" with "CancellationError/F005 unchanged there" — true
   as of 8d, false after c1a552f (that exit itself now runs a closed form,
   not the series) and false after 4318dab (CancellationError deleted).
3. ENGINE HARDENING (Build 7a) point (2), a "cross-parity fallback... routes
   CancellationError refusals" — historical narrative, left the sentence
   intact (matches this doc's own precedent of preserving build history) but
   appended a SUPERSEDED-by-8d/CancellationError-retirement note rather than
   deleting it.
4. Two "named refusal vocabulary" lists (LensedPosterior's refusal net, in
   two different rows) both still listed `CancellationError` as live —
   deleted from both.
5. ONE-HOME PREDICATE / UNIFORM-ASYMPTOTIC SERVING paragraph's `select_branch`
   description ("resolved AND L > L_MAX") was missing the third (eta) leg
   4318dab added — this is the SAME select_branch SPEC already documents
   elsewhere, so the omission would have made SPEC self-inconsistent, not
   just outdated. Added the eta leg + FINDINGS F031 pointer in both the
   serving-ladder sentence and the F029-tail sentence (F031 is F029's fix).
   Also noted the saddle now passes `eta = inf` alongside its existing
   infinite cancellation exponent (F031 is positive-parity-only evidence).

VERIFIED, NOT edited: `MAX_ORDER` is genuinely vestigial now (threaded as a
parameter default through `F_op`/`F_op_grid`/`_positive_parity_grid`/
`ChangRefsdalChannels.__init__` but never referenced inside
`_positive_parity_grid`'s body — grep confirms zero use past the signature).
SPEC.md never named `MAX_ORDER` directly (it's implementation-level), so
nothing to fix there; noted the fact in the spec_changelog fragment instead
of inventing a SPEC sentence about it, per the "SPEC never described this
criterion in the first place -> no staleness to manufacture" house rule.

FINDINGS.md (canonical, hand-maintained, not fragment-generated) had its own
staleness: F031's own body still concluded "**NOT implemented, deliberately**"
even though 4318dab's commit message says it WAS implemented that same
commit, with the SAME measured numbers (p90 1.17 -> 7.65e-5) the "not
implemented" paragraph cites as a future possibility. This is a genuine
self-contradiction inside one finding entry, not a downstream-sync gap —
4318dab touched FINDINGS.md (197 insertions) but that diff evidently went to
extending F030 (the GLoW root-cause investigation, confirmed by reading it —
long GLoW/tmin/Fw-NaN narrative) and never circled back to close F031's own
verdict sentence. Fixed the verdict paragraph in place (IMPLEMENTED, cites
4318dab, keeps the still-true caveats: positive-parity only, no oracle above
the Schwinger ceiling).

NOT touched, flagged only: `.claude/spec/todo.d/tests_consolidate_duplicate_
routing_pins.md`'s table ("select_branch routing | 16 | 6") is now an
overcount — 4318dab's own commit message says duplicate `select_branch`
routing pins were deleted from schwinger/airy_fold/levers, and I found
"DELETED (one-home consolidation)" pointer-comments confirming this in
`test_lensing_schwinger.py` and `test_lensing_airy_fold.py`. Did not edit the
fragment: no clean before/after count without a slower per-file audit, and
per house rule a multi-part TODO stays open until every part (also
`SchwingerCertificationError` and `W_CEILING_SCHWINGER` pin counts, untouched)
finishes — the stale COUNT doesn't change the OPEN status, so it's a nice-to-
fix not a must-fix. Next Librarian: re-count before touching.

git mechanics: `scripts/render_fragments.py` bumped spec_version 0.25.0 ->
0.26.0 (minor, per my own spec_changelog fragment's `bump: minor`) and left
the usual stray `.claude/tidy_advisory.json` diff (commit-tracking metadata
racing ahead to 4318dab) — reverted with `git checkout --` per established
pattern, not committed. `cogwheel/tests/test_lensing_batched_operator.py`
arrived ALREADY STAGED (index differs from HEAD) at session start — not mine,
didn't touch it, committed my own files via `git commit -- <explicit paths>`
so its staged state survives untouched in the index for whoever owns it.
`docs/source/**` had zero hits for any of these terms — no Sphinx rebuild
needed this session (nothing under docs/source touched).
