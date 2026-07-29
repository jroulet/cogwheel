## 2026-07-29 (later) post-commit sync — fold arm eta fence / saddle eta leg / max_order removal (commits 285f6cc..d4ee4cb)

Scope: 7 pending commits. Only cf1267f had a real cogwheel/ production diff
(_airy_fold.py +15, channels.py -38, operator.py -140, surrogate.py -14);
the rest were test-only or FINDINGS-only (already committed by the feature
author, per house rule "verify-only is correct outcome" — checked, no doc
staleness from those).

STALENESS FOUND AND FIXED in SPEC.md's single giant "Microlensing engine"
table row (3 targeted literal replace_content edits):
1. The saddle branch of `select_branch` was described as passing
   `eta = inf` with "whether the saddle needs its own eta floor" left as
   an OPEN, UNMEASURED question. FALSE after 7b775a1/F034: `_saddle_grid`
   now measures eta via `nearest_caustic_point` and passes it through —
   the eta leg is live on BOTH parities (positive parity F031, saddle
   F034). Rewrote with F034's numbers (p90 8.95e-1 -> 4.54e-3, worst case
   484x, 15% of resolved draws).
2. SPEC.md's bullet (a) about F028 (fold arm 60%-267% wrong) had NO
   closing note — the arm now carries a caustic-relative admission fence
   (`_ETA_MAX_FOLD = 0.3` in `_airy_fold.py`, a LITERAL not an import of
   `ETA_MIN_GEOMETRIC` because `operator` imports `_airy_fold`). Added the
   fence + F032 (independent GLoW confirmation, 63-64% wrong) + F033 (why
   the fence, not a `b4` amplitude fix, is the permanent treatment: the
   residual is the CUBIC NORMAL FORM's own O(eta) truncation, not `q=0`).
3. Same fold-arm fence added to the `_airy_fold.py` architecture sentence
   itself (the None-fall-through contract now has two triggers: outside
   the fence, or otherwise uncertifiable).

VERIFIED, NOT touched: `max_order`/`MAX_ORDER` was never named in SPEC.md
(confirmed by grep, same as last session) despite being fully removed from
cogwheel/ (`F_op`, `F_op_grid`, `ChangRefsdalChannels.__init__`,
`_positive_parity_grid`, `surrogate.from_engine`/`from_lobe_engine`, plus
orphaned `_MIN_ORDER`/`_CONSECUTIVE_SMALL`/`_SERIES_TOLERANCE` module
globals) — so no SPEC sentence needed correcting, only a changelog entry
(public API break, the notable user-facing item this round).
DATA_CONTRACTS.yaml: zero max_order hits, nothing to touch (no disk
artifact). docs/source: zero hits for any of these terms — no Sphinx
rebuild needed. TODO.md / todo.d/lensing_fold_arm_serves_wrong_values.md:
already correctly rewritten by the feature commit itself (95fa3f8) to
describe the CURRENT fence and point at F033's "don't derive b4" finding
— verify-only, matches the SPEC `[→ spec]` tag I've now closed.

NOT FIXED, FLAGGED ONLY (out of scope — code file, not doc):
`operator.py`'s OWN `select_branch` docstring (its Notes section) still
says "The macro saddle passes `inf` deliberately... whether the saddle
needs its own eta floor is OPEN" — this is now stale INSIDE the code
itself post-7b775a1, which changed `_saddle_grid`'s call site but never
touched `select_branch`'s docstring. Librarian scope is docs, not code
files (hard rule); flagging for the Coder/Inspector to fix the docstring
to match the `_saddle_grid` behavior it documents.

NOT FIXED, FLAGGED ONLY (pre-existing, predates this window): the
"Build 8b-levers" historical paragraph in SPEC.md's same giant row still
describes `operator._fused_contraction`, `half_sum`, `_SERIES_TOLERANCE`
as a "patchable module global", and
`test_lensing_fast_path.py::OperatorFusionFalsificationTestCase` as live
mechanisms. Checked: `_fused_contraction` and
`OperatorFusionFalsificationTestCase` were ALREADY GONE from the codebase
at eff1de7 (before this session's window) — only `_SERIES_TOLERANCE`
newly died in this window (cf1267f). This paragraph is historical
Build-8b-levers narrative (house convention: preserve with a SUPERSEDED
note rather than delete) but is now a compound dead-reference case
spanning two sessions; didn't touch it this round — flag for next
Librarian, needs a proper SUPERSEDED annotation, not a quick patch.

git mechanics: `render_fragments.py` bumped spec_version 0.26.0 -> 0.27.0
(minor). Confirmed AGAIN the known out-of-order-versioning quirk: my
fragment `2026-07-29_fold_arm_fence_saddle_eta.md` got assigned 0.26.0
while the PRE-EXISTING `2026-07-29_operator_series_retired.md` (same
date, alphabetically later filename) got 0.27.0 and renders ABOVE mine —
bump-by-filename-alphabetical, not by content chronology or mtime. Left
the stray `.claude/tidy_advisory.json` diff (commit-tracking metadata
racing ahead) reverted via `git checkout --`, not committed — same
pattern as every prior session.
