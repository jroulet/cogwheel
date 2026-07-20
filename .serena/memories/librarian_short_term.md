# Librarian Short-Term Observations (2026-07-20)

Scope: post-build doc-sync, Build 5/6/7a lensing cycle (commits 3b3ebdb,
88e5386, 83d75dc; last_commit baseline b2d80a0). Dreamer flagged
INS-5-DOC-1/INS-4-DOC-1/INS-1-001 (SPEC rows for the marginalized lensed
likelihood/prior classes) as "unconsumed."

Finding: **already resolved, no-op run.** `git show --stat 3b3ebdb`
shows `.claude/spec/SPEC.md` + `spec_changelog.d/2026-07-18_lensed-marginalized.md`
were touched in the SAME commit that added marginalized_likelihood.py —
someone (Architect/driver hand-fix, per crew note "SPEC engine row was
hand-updated through Build 7a") already consumed the INS-5-DOC-1 fix
before this Librarian run. Verified directly: SPEC.md's "Microlensed
sampling layer" row (~line 55) names `LensedMarginalizedExtrinsicLikelihood`,
`LensedMarginalizedExtrinsicIASPrior`, the d_app/d_L=d_app*sqrt(mu_macro)
F009 note, and `cogwheel/lensing/marginalized_likelihood.py` in the
module-path column — exactly what the escalation's suggested_fix asked
for. Do not re-touch; verify-only was correct per the task framing.

- SPEC.md engine row (~line 53) also already covers Build 6/7a:
  negative-parity/macro-saddle branch, `_schwinger.py`, F011-F013,
  cross-parity Schwinger fallback (F015). Verified F001-F015 all
  resolve in FINDINGS.md; SPEC.md cites F005/F008/F009/F010/F011/F012/
  F013/F015 (F014 uncited — expected, it's a standalone convention note,
  not referenced elsewhere either).
- `docs/source/api.rst` uses bare `autosummary :recursive:` over
  `cogwheel` — confirmed (again) this auto-covers new subpackages
  (`cogwheel.lensing.marginalized_likelihood`, `.chang_refsdal._schwinger`,
  etc.) with zero manual entries needed. Reconfirms the prior long-term
  note; still true after 3 more lensing builds.
- `docs/source/overview.rst` "Microlensing engine" section: NOT stale.
  It describes `ChangRefsdalChannels` (channels.py, "the public entry
  point") as positive-parity-macro-images-only — this is STILL TRUE at
  the channel-layer public API even after Build 6/7a extended
  geometry/operator to accept saddles internally: verified in
  `cogwheel/lensing/chang_refsdal/channels.py` (~line 900-908) that
  `ChangRefsdalChannels` still raises `LensDomainError` on saddles by
  name, pending the "Build 7 saddle channel layer." The narrative also
  omits the sampling-layer/marginalized classes entirely, but that
  matches the doc's existing pattern (Prior/Likelihood sections there
  are generic, don't enumerate every concrete subclass either) — not
  new staleness introduced by this cycle. Left unedited; flag for
  re-check once the saddle channel layer (Build 7+) actually lands —
  THAT will flip the positive-parity-only claim to false.
- `docs/source/index.rst`: no module enumeration, nothing lensing-
  specific to check.
- `.claude/spec/DATA_CONTRACTS.yaml` has zero lensing entries by design
  (samples.feather column names unchanged per INS-5-DOC-1's own
  suggested_fix) — confirmed still empty, correctly so.
- `scripts/sync_derived_docs.py` ran clean (5 checks, all OK) before
  triage.

Net effect: zero files edited this run. All three Dreamer-flagged doc
gaps were dead by the time this Librarian ran. No sphinx rebuild
triggered (docs/source/ untouched).

Fragile cross-reference to watch: overview.rst's "positive-parity macro
images only" claim for the engine is TWO-LAYERED now (geometry/operator
support saddles; channels.py/waveform.py do not) — a future Librarian
must re-read channels.py's actual refusal code, not just SPEC.md prose,
before editing that sentence, since SPEC's engine-row prose could
describe geometry/operator (which now DO support saddles) in a way
that superficially looks like it contradicts overview.rst's channel-
layer claim. They're both correct but at different layers.
