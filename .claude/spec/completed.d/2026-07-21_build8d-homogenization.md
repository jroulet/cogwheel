---
date: 2026-07-21
section: likelihood
---
### Build 8d — homogenization (complete)

Schwinger = the exact wave evaluator on both parities (WP1); legacy
demoted to gamma'==0 exit + legacy_operator_oracle; corner census
(WP3) measured the w>60 non-geometric corner (~25% upper bound of
prior draws) and the gamma'=0 fraction (~0). Owner rulings: pure
homogenization shipped (corner refuses until 8e; a per-band legacy
revival was reverted); slow test tiers never run in-build (enforced:
env-pinned in the SDK, COGWHEEL_BRUTE_ACCURACY gating, post_build_
sweeps.sh). Re-baselines by independent test dev: witnessed contract
flips <= 3.6e-14, refusal-vocab updates (_WAVE_REFUSALS), timing
gates re-tuned, zero genuine regressions, F015 net verified. F019.
Pipeline crashed at the revision-coder spawn (argv, root-caused to
spec inlining — de-inlined; port items 9-13); driver hand-finished.
