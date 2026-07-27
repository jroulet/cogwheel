# Librarian Short-Term Observations (2026-07-27, second pass this date)

Scope: cleared the 14-commit backlog (d84a2d5..d0dc6da, ghost delay-frame
repair through the ghost gate re-key and geometry-partition perf commit).
Per driver's explicit scope: cogwheel/lensing/** and cogwheel/tests/**
production/test code untouched (read-only); test-only commits (a4385b2,
3c107d4, 1faca34, 56b755c, edf8d48, 2cd23b3) and agent-infra-only commits
(d84a2d5, 1e60697) skip-entirely per the triage table — verified each by
reading its changed_files list in sync_issues.json rather than re-deriving.

## What was actually stale (two SPEC.md sentences, both self-inflicted by
## a WITHIN-BACKLOG code change outpacing the doc entry that described it)

1. **Delay frame authority drifted one commit after being documented.**
   6658e6d added the "Lensing delay frame" Conventions entry naming
   `_frame_t_min(source, matrix)` as the single authoritative frame-origin
   expression. The VERY NEXT lensing commit, 74c1d55, refactored this:
   `_frame_delays` became THE authoritative construction (returns
   `(images, absolute_delays, t_min)` together, so partition builders don't
   re-solve the image quartic) and `_frame_t_min` was demoted to "a thin
   accessor over `_frame_delays`" per its own docstring. 74c1d55 touched no
   doc files (see its changed_files list — channels.py/likelihood.py/tests
   only), so the SPEC.md entry silently went stale one commit after
   landing. Pattern to watch: when a build's SPEC.md entry documents a
   just-added function by name, and a LATER commit in the same backlog
   refactors that function's role, the entry doesn't self-correct — check
   the CURRENT docstring of any function SPEC.md cites by name against
   what SPEC.md says about it, don't trust that citing it once means it's
   still accurate three commits later.

2. **Distance convention flip (4ffbde5) never reached SPEC.md at all.**
   4ffbde5 rewrote `cogwheel/lensing/waveform.py` as "the single
   authoritative statement" that `d_luminosity` is PHYSICAL luminosity
   distance on both routes (LensedIASPrior's d_hat->d_luminosity, and the
   marginalized coherent-score blob column) — flipping an OLDER SPEC.md
   sentence (predating this backlog, likely from the original F009 build)
   that said `d_luminosity` was the APPARENT distance requiring a
   post-analysis `sqrt(mu_macro)` rescale. 4ffbde5's changed_files list
   (CHANGELOG.md, changelog.d, marginalized_likelihood.py, prior.py,
   waveform.py, tests) never included SPEC.md — this was a genuine gap,
   not an artifact of my triage. Fixed by rewriting the one sentence in
   the "Microlensed sampling layer" row's Marginalized-path paragraph.
   FINDINGS.md F009 was checked and does NOT restate the apparent/physical
   distinction (it's about the F(w->0)=sqrt(mu_macro) physics only) — left
   untouched, correctly.

Both fixes recorded in one fragment
`spec_changelog.d/2026-07-27_librarian_sync_backlog.md` (bump: patch — a
correction to already-recorded content, not new capability), rendered via
`render_fragments.py` -> spec_version 0.21.0 -> 0.21.1, last_updated stays
2026-07-27 (the newest-dated fragment already led the two-tier sort, so
this is the first pass in a while where last_updated genuinely reflects
the latest change instead of being stuck — see long-term memory's
"last_updated never advances" note; that failure mode did NOT recur here
because 2026-07-27 fragments now dominate the dated tier).

## Confirmed NOT stale (checked, not skipped blind)
- `docs/source/api.rst`: `:recursive:` autosummary over bare `cogwheel`
  confirmed still in place — `_born.py` (private) and all of 87643d7/
  d0dc6da's channels.py changes need no manual api.rst entry.
- `docs/source/*`: zero grep hits for d_luminosity/ghost/Born/delay
  frame/mu_macro/apparent — the Sphinx narrative sits above this
  implementation layer, consistent with prior passes. No Sphinx rebuild
  needed (nothing under docs/source/ touched).
- `DATA_CONTRACTS.yaml`: `ChangRefsdalGeometryPartition` (which gained
  `t_min`/`images` fields in 74c1d55/d0dc6da) is an in-memory dataclass,
  never serialized to disk — confirmed via its docstring/attributes
  before deciding it needs no contract entry. DATA_CONTRACTS.yaml is for
  disk artifacts; don't conflate "gained new fields" with "needs a
  contract" without checking whether the object round-trips through disk.
- `sync_derived_docs.py`'s only complaint (4 warnings) was the familiar
  test-file-only `lens_amplification_surrogate` consumer gap (via
  `LensAmplificationSurrogate.load` in test_lensing_surrogate.py) — same
  pre-existing pattern as prior passes, left off per the established
  test-file-only-consumers convention (see long-term memory).
- Ghost gate re-key (87643d7, DECAY -> GEOMETRIC SEPARATION criterion,
  `_GHOST_SEPARATION_MIN = 0.7`): grepped SPEC.md for "decay"/"admission"/
  "w_min"/"Im tau_c" — the only hits were inside the delay-frame
  Conventions entry (about frame mixing, not the admission criterion
  itself). SPEC.md's Conventions section never described the gate
  criterion's specific formula in the first place (that detail lives only
  in the channels.py module docstring/constants), so there was nothing to
  flip here — don't manufacture a gate-criterion sentence that wasn't
  there before; the module docstring is the right home for that level of
  detail, not SPEC.md.

## Reconfirmed from long-term memory (both held on this pass)
- Two-tier SPEC_CHANGELOG sort (dated fragments always outrank undated
  ones): still true, still a "flag don't fix" — not touched this pass.
- Fenced-dir safety: `git status --short` before AND after my edits
  showed only .claude/agent_state/*.json, .claude/tidy_advisory.json, and
  several .serena/memories/*_short_term.md as pre-existing dirty state
  from concurrent agents — left entirely alone, not part of my diff, not
  staged in my commit.
