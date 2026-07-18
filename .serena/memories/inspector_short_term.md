# Inspector Short-Term Observations

## 2026-07-18 — Build 4 REVIEW: sampling layer (lens prior + posterior + pickle plumbing)

Scope: uncommitted worktree /home/tejaswi/Work/cogwheel-claude-dev
(HEAD 93825be; Serena root == worktree). Changed code:
- cogwheel/lensing/prior.py (NEW, 256 lines) — WP1/WP2: reduced-coordinate
  lens subpriors + registered LensedIASPrior.
- cogwheel/lensing/posterior.py (NEW, 74 lines) — WP2: LensedPosterior
  refusal net.
- cogwheel/lensing/likelihood.py (+22) — WP3: __getstate__/__setstate__
  drop _fid_cache, keep _force_direct.
- cogwheel/lensing/__init__.py (+5) — export LensedIASPrior/LensedPosterior.
- cogwheel/tests/test_lensing_prior.py (NEW, 1202 lines) — C1..C7 +
  SelfFalsification.

VERDICT: PASS. No bug/design/impl findings introduced by this build.

Verification performed this session:
- New suite: 27 passed + 1 xfailed (62 s). xfail = documented aspirational
  0.90 finite-fraction (prior box overlaps gamma~0.5 cancellation band; ~59%
  refused-to-inf is a documented prior-width property, not a bug).
- Regression: test_lensing_likelihood + test_lensing_ratio_layer = 47 passed
  + 1 xfailed (89 s). No regression from __getstate__/__setstate__ or the
  __init__ export change.
- WP3 pickle round-trip probed directly: __getstate__ drops _fid_cache,
  keeps _force_direct + other dict entries; __setstate__ resets _fid_cache={}
  and preserves _force_direct; real pickle.loads(dumps) confirms. No base
  class defines __getstate__/__setstate__ (grep), so override is safe (no
  super() needed). JSON path unaffected (get_init_dict uses __init__ sig).
- Refusal net site verified: Posterior.lnposterior -> self.lnposterior_
  pardic_and_metadata()[0] (dynamic dispatch) AND sampling.py wraps
  self.posterior.lnposterior_pardic_and_metadata via prior.unfold_apply.
  So both scalar and folded/sampler paths route through the override =>
  binding constraint #2 (no unswallowed refusal under proposals) satisfied
  at exactly the boundary site. except clause is specific-named
  (LensDomainError, CancellationError), not bare. Transform recompute in
  the handler is pure coordinate map (cannot raise engine refusal).
- prior.standard_params (25) == likelihood.params (instance property =
  sorted(wfg.params | _LENS_PARAMS)) — harness asserts at runtime; import
  probe shows all 7 lens params present.
- Mixins: UniformReducedShearPrior(UniformPriorMixin, IdentityTransformMixin,
  Prior). IdentityTransformMixin subclasses UnitJacobianMixin => identity
  transform + 0 Jacobian + standard_params=['gamma']. Plan named
  UnitJacobianMixin; IdentityTransformMixin includes it (cleaner, correct).
- _source_scale cap (min(307/m, 3.0)): corner product w*sqrt(s) = const 55
  in non-cap region (by _Y_SCALE=307 design) and smaller in cap region; C3
  10^4-pt sweep confirms <=58. Jacobian uses same scale in transform+inverse
  so round-trip (C1) and FD-Jacobian (C2) match regardless of cap branch.
- Test oracles non-circular: C4a/C5 build fresh ChangRefsdalChannels per
  config (astroid symmetry / professor mass-sheet closed form as oracle,
  not pipeline reuse — F002 respected). C6 mutation (patch module-global
  CancellationError) turns -inf gate red. SelfFalsification proves C1/C5
  gates + anti-vacuity tearDown can go red.

APPROVED DEVIATION (NOT a finding): d_app deferral. Binding constraint #1
says "sample d_app = d_L/sqrt(mu_macro)". Build samples physical
d_luminosity (reuses IASPrior UniformLuminosityVolumePrior); a lens-aware
d_app subprior is DEFERRED to Build 5. This is EXPLICITLY approved and
documented in build4_plan_approved.json ("Distance stays physical ...
DEFERRED to Build 5"; ASSUMPTION 1) and in the LensedIASPrior docstring.
kappa is fixed=0 (never sampled) — constraint #1's build-killer teeth
honored. Not a Coder defect.

CARRIED FORWARD (Librarian doc-sync, NOT Coder defects):
- INS-4-DOC-1 [SPEC.md]: no prior-layer row for LensedIASPrior/
  LensedPosterior/reduced coordinates yet. Plan ASSUMPTION 3 explicitly
  defers the SPEC row to post-gate doc-sync (not a Coder WP). Add a
  prior/coordinates row + note d_app-deferred-to-Build-5.
- INS-1-001 (Build 3g, still open): SPEC line ~55 warm lnlike ~0.3 s/eval
  + "2D surrogate-table decision"; ratio layer (9.8 ms, _fid_cache) not
  reflected. Doc-sync.
- Program todo microlensed-PE item 3 fragment: retire only after post-build
  injection-recovery validation (driver, detached) — note, don't retire.

KB reinforcement: Build runs entirely in sibling worktree
cogwheel-claude-dev; `cd` to /home/tejaswi/Work/cogwheel is blocked by the
serena-shell hook (git-only exception needs no cd). Use Bash from the
worktree cwd directly.
