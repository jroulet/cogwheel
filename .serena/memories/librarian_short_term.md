# Librarian Short-Term Observations

## 2026-07-17 post-commit audit (16 queued commits, 21243c7..398a57a; HEAD e8f2c58)

Scope: Build 2 lensing modules (`cogwheel/lensing/waveform.py`,
`cogwheel/lensing/likelihood.py`), the Build 2c channel-switch fix, the
Build 2d macro-limit certification, plus unrelated SDK/gate/orchestrator
commits mixed into the same queue.

- Verified the queue against reality: `git diff --stat 21243c7~1..398a57a`
  matches the union of `changed_files` across all 16 entries. Note for next
  time — the queue is NOT every commit in that range; it skips several
  intervening ones (`aea168b`, `f9472d4`, `8232d02`, `5974212`) that a prior
  run already synced (their messages are `docs:`/`chore:` self-closing).
  That's expected, not a gap.
- Result: **no stale doc surfaces found** — this was a no-op sync run.
  Checked and confirmed already in sync:
  - `SPEC.md` — Build 2 module row (waveform.py/likelihood.py) present and
    matches the shipped API; no `kernel_subsamples` literal needed there
    (implementation detail, not part of the documented contract surface).
  - `DATA_CONTRACTS.yaml` — correctly has no lensing entries: both classes
    are in-memory `JSONMixin` objects, no on-disk artifact to register.
  - `docs/source/overview.rst` — Build 2 paragraph already present
    (`LensedWaveformGenerator` + `LensedRelativeBinningLikelihood`, `F(w)`
    convention, positive-parity-only guard).
  - `docs/source/api.rst` — `:recursive:` autosummary over bare `cogwheel`
    already covers `cogwheel.lensing.*`; reconfirms the standing note that
    no manual entry is needed for new subpackages under it.
  - `index.rst` / `crash_course.rst` / `installation.rst` — no lensing
    references needed yet (experimental, no tutorial surface).
  - `FINDINGS.md` — F006 correctly marked SUPERSEDED by F008; F008/F009
    cross-references resolve.
  - `todo.d/2026-07-16_lensing-program.md` — correctly still open (frames
    a 3-part program; Build 3 is pending) — did not move to completed.d.
    Consistent with the standing rule: don't complete a multi-part program
    fragment when only some parts landed.
  - `python scripts/render_fragments.py` and `sync_derived_docs.py --check`
    both reported clean with zero diffs — every canonical file already
    reflects every fragment in the queue.
- Did not touch `todo.d/likelihood_standard-rb-zero-noise-floor.md` or
  `todo.d/sdk_brief-depth-guards.md` — explicitly out of scope, still open.

### Surprise (flagged to caller, not fixed — outside Librarian's ownership)

`SPEC_CHANGELOG.md` version numbers are **not chronological**: `0.1.0`
describes Build 2 (waveform+likelihood — logically the later work) while
`0.1.1`/`0.1.2` describe Build 1b (engine complete — logically earlier) and
`0.2.0` describes the Build-1 salvage foundation (earliest of all). Root
cause: `render_fragments.py` assigns versions by processing
`spec_changelog.d/` fragments in **alphabetical filename order** and
stacking each fragment's `bump:` on top of a running version — not by the
fragment's own `<date>` prefix or the logical build sequence.
`lensing-build2-waveform-likelihood.md` sorts before
`lensing-engine-complete.md` and `lensing-foundation.md` alphabetically,
even though it documents later work. Every individual entry's prose is
correct; only the version-number sequence reads out of order. Left alone:
fixing it means either renumbering fragments (risks breaking any existing
reference to `0.1.0`/`0.2.0` etc.) or changing render-script ordering
behavior — both are code/process changes outside "sync existing
information across surfaces," and more Inspector's call (spec-as-checkable-
invariant) than mine.
