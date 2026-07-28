# Librarian Short-Term Observations — 2026-07-28 (post-commit for 8901b0b, 04f9f5c)

Scope: sync_issues.json listed 8901b0b (telescoping-bound test fix / Tidy
repair / retire branch-vs-HEAD apparatus) and 04f9f5c (macro-saddle
per-lobe interior surrogate charts, SPEC.md -> 0.22.0) as pending. The
task prompt called them "(3d8fd8f, 04f9f5c)" but the actual trigger file
listed 8901b0b + 04f9f5c — 3d8fd8f (Dreamer memory consolidation) touches
no doc-relevant files and was correctly absent from the trigger; treated
the trigger file as authoritative, not the prompt's paraphrase.

What I fixed:
- Added `changelog.d/2026-07-28_saddle_lobe_serve.md` for 04f9f5c. EVERY
  prior lensing-surrogate feature build (8a/8b/8c/8d/8e/8f/8g/8h-b4/
  farfield-envelope-v2) has a changelog.d entry describing the change in
  prose synthesized from SPEC.md; 04f9f5c (new public `LobeInteriorChart`
  + `from_lobe_engine`, closes the saddle-interior coverage gap) fit that
  precedent exactly and had none despite touching everything else
  (SPEC.md, TODO.md, tests). Ran render_fragments.py after.

What I verified and left alone (no action needed):
- SPEC.md's 04f9f5c diff is a single-row rewrite (Microlensed sampling
  layer) plus the version bump — nothing else in SPEC.md changed, so no
  other SPEC-internal cross-references went stale.
- DATA_CONTRACTS.yaml: Inspector's INS-3-001 "no schema bump" judgement
  HOLDS — the `lens_amplification_surrogate` description already reads
  "per-chart coefficient/knot arrays" generically; confirmed no chart-kind
  enumeration exists there to go stale.
- api.rst: `:recursive:` autosummary over bare `cogwheel` (still true) —
  new class/classmethod in an existing module (`surrogate.py`) needs no
  manual entry. Reconfirms the standing memory note.
- overview.rst / crash_course.rst / index.rst: lensing is not mentioned in
  crash_course.rst or index.rst at all (intentionally out of the
  user-facing narrative — experimental, off-by-default). overview.rst's
  "Microlensing engine" section is the only Sphinx mention.
- 8901b0b's non-test .py diffs (_pearcey_table.py, _schwinger.py,
  posterior.py, ppgo_map.py, surrogate_training.py) are PURE Tidy
  line-wrap reflow, zero logic change — confirmed via full diff, not just
  the commit message. The actual telescoping-bound fix lives entirely in
  test files (test_lensing_farfield_envelope.py,
  test_lensing_ppgo_bandsplit.py) and is already narrated in
  `completed.d/2026-07-28_telescoping-conditioning-bound.md`
  (self-synced by the commit). Considered promoting it to a FINDINGS.md
  entry (it's a real condition-number-based tolerance derivation, exactly
  FINDINGS.md's stated subject matter) but left it as completed.d-only:
  it reads as test-bound bookkeeping for one xfail, not a reusable trap,
  and the commit author already chose completed.d as the record — did not
  second-guess a surface choice that was itself already made in-commit.

SURPRISE — found, but NOT fixed (predates both queued commits by ~15
builds, so out of THIS post-commit's scope; flagging for a future pass):
overview.rst's "Microlensing engine" section still says the two
higher-level entry points (LensedWaveformGenerator,
LensedRelativeBinningLikelihood) "support positive-parity macro images
only: a configuration with `1 - kappa <= |gamma|` raises ... rather than
returning a degraded result." This has been false since the negative-
parity/macro-saddle branch landed (Build 6/7, changelog.d
2026-07-19_saddle-branch.md / 2026-07-20_saddle-integration-7b.md) — both
parities have been supported for ~15 SPEC versions. Also never mentions
the surrogate layer (8a-8h) at all. Prior "backlog clear" sync passes
(aff725f and earlier) missed this too — it is NOT new staleness from
8901b0b/04f9f5c, so I did not rewrite it here (would be a substantial,
judgment-heavy content addition, better done as its own scoped pass
rather than folded into a two-commit post-commit sync). Next Librarian
run (or a dedicated doc-sync invocation) should pick this up explicitly.

Mechanical notes:
- `sync_derived_docs.py` run 1 left the now-familiar stray
  `.claude/tidy_advisory.json` diff (commit-pointer rewrite as a side
  effect) — reverted with `git checkout --`, not committed. Confirms the
  standing memory note, now observed a second time on a different script
  (previously only seen from render_fragments.py).
- `sync_derived_docs.py` also flagged 4 test-only consumers of
  `lens_amplification_surrogate` (via `LensAmplificationSurrogate.load`)
  as "not in DATA_CONTRACTS.yaml — add it or confirm transient". Per
  standing convention (consumer lists are production-only), these
  correctly stay off the list — no action.
