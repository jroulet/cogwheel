# Inspector Short-Term Observations

## 2026-07-18 — Build 3g ratio layer RE-REVIEW #2 (lensing/likelihood.py)

Scope: uncommitted working tree in WORKTREE /home/tejaswi/Work/cogwheel-claude-dev
(HEAD 0adcfb7; Serena project root == worktree). Changed: cogwheel/lensing/
likelihood.py (`git diff --stat` = 606 insertions / 18 deletions — BYTE-IDENTICAL
to the prior 3g review; the "+624" in the task header was git-status churn
counting) + untracked new suite cogwheel/tests/test_lensing_ratio_layer.py
(1099 lines). Plan named test_lensing_fast_path.py but a dedicated new module was
shipped instead — benign plan deviation (KB: new module vs editing existing suite
is benign).

VERDICT: PASS. No bug/design/impl findings introduced by this build.

Re-verification this session (byte-identical diff still re-run per KB rule):
- import probe OK (_fiducial_key, _FiducialEnvelope present).
- 18 passed (test_lensing_ratio_layer, 37.7s).
- 46 passed + 1 xfailed (fast_path + likelihood, 85.7s). Original tolerances.
- No njit in likelihood.py (grep: only 2 doc-comment "no njit is introduced"
  mentions). Pure numpy ratio path — F010 N/A.
- SelfFalsificationTestCase GENUINE: (a) spurious exp(1j*w*eps) carrier ->
  ENVELOPE_IDENTITY_RTOL gate fires; (b) mock.patch.object(likelihood_module,
  'reconstruct_from_envelope', *1.5) -> RB-vs-brute gate fires (module-level
  patch works BECAUSE reconstruction is not njit); (c) anti-vacuity tearDown
  fails on n_checks==0. All three pass (gates fire).

Correctness (re-derived, unchanged from prior session — still holds):
- _fiducial_key snaps gamma/beta/kappa/y1/y2, keeps m_lens_msun & z_lens EXACT
  => candidate & fiducial dimensionless_frequency grids coincide (same instance
  _kernel_dense_f) => fiducial spline support covers candidate dense_w => no
  extrapolation in _FiducialEnvelope.envelope. _lens_from_key exact inverse.
- Lattice-point identity: lens==snapped fiducial => dtau_c=0, E_cand==E_fid at
  engine, seed geomspace identical => rho==1 at seed nodes => no refinement =>
  envelope_dense == direct candidate spline. Machine-eps identity (test ~1e-11,
  gated 1e-9).
- Ratio reconstruction uses CANDIDATE partition geometry (tau_c_cand, delays,
  saddle/switch) via _kernels_from_dense_envelope(..., partition_cand); fiducial
  only supplies E_fid & dtau_c carrier which cancel. The "using fiducial tau_c
  would be a correctness error" trap is correctly avoided.
- Refusal symmetry: ONLY _get_or_build_fiducial wrapped in try/except
  (LensDomainError, CancellationError) -> fallback to direct (a refusing SNAPPED
  fiducial must not veto an in-domain candidate). Candidate seed eval + LOO
  refinement propagate refusals unswallowed on ratio + direct. RefusalSymmetry
  suite covers macro-saddle + uncertifiable-branch.
- Guards: image-count mismatch (real_mask.sum) and health floor
  (min|E_fid|/max|E_fid| < _ENVELOPE_HEALTH_FLOOR=0.01) both fall back to direct.
- __init__ adds _fid_cache={} and _force_direct=False as NON-ctor attrs =>
  JSONMixin get_init_dict (signature(__init__)) does not serialize them => no
  DATA_CONTRACTS change (contracts cover serialized artifacts only).
- WP1 refactor: _refine_envelope_grid / _kernels_from_dense_envelope /
  _image_delays / _reduce_dense_kernels extracted; behavior-preserving (46
  passed confirms). _amplification_coefficients_direct is the old body renamed.

Carried-forward (NOT resolved, NOT Coder defects — recorded per hard rule):
- INS-1-001 [SPEC.md line 55, Librarian doc-sync phase]: 'Microlensed waveform &
  likelihood' row STILL says warm lnlike ~0.3 s/eval and defers few-ms to a
  '2D surrogate-table decision'; no mention of ratio layer / _fid_cache /
  _force_direct. Plan EXPLICITLY defers doc-sync post-gate => expected, not a
  Coder defect. Reconcile SPEC row + timing figure + FINDINGS ratio-layer
  addendum in doc-sync.
- INS-1-002 [test_lensing_ratio_layer.py line 156]: ENVELOPE_IDENTITY_RTOL=1e-9
  vs brief's 1e-13 (measured ~1e-11 cross-grid floor). Honestly documented; 7
  orders below _LOO_STOP=4e-3; lnlike identity still gated 1e-9 (LNLIKE_IDENTITY
  _ATOL). Accepted, non-blocking.

KB reinforcement: Coder works in SIBLING WORKTREE (cogwheel-claude-dev); main
tree /home/tejaswi/Work/cogwheel shows only agent_state/memory churn while the
worktree carries the whole build. Always git-status BOTH.
