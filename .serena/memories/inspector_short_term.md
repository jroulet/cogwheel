# Inspector Short-Term Observations

## 2026-07-18 — Build 5 RE-REVIEW #7 (marginalized lensed likelihood)

Scope: uncommitted worktree (cogwheel-claude-dev). Code diff = lensing/__init__.py
(+exports LensedMarginalizedExtrinsicIASPrior + LensedMarginalizedExtrinsic
Likelihood) + lensing/prior.py (+LensedMarginalizedExtrinsicIASPrior class,
import IntrinsicIASPrior); two untracked new files
(marginalized_likelihood.py, tests/test_lensing_marginalized_likelihood.py).
Byte-identical to the prior SIX Build-5 reviews. Re-verified FRESH this pass.

VERDICT: ISSUES (one carried doc-sync finding INS-5-DOC-1 only; code green).

### Re-verified this pass
- `git diff SPEC.md DATA_CONTRACTS.yaml` = EMPTY (both unchanged) -> INS-5-DOC-1
  STILL OPEN (Librarian-owned, reserved commit-time). NOT a code defect.
- Base `_get_dh_hh_timeshift` (marginalized_extrinsic.py L346-366) read directly;
  override folds F into data (h_lensed=amplification*h_mpb, reused `_d_h_weights`)
  and |F|^2 into norm_weight (broadcast (1,1,b) over `_h_h_weights` mdb; unlensed
  h_mpb in mode-pair einsum). Structure matches base exactly.
- Engine symbols confirmed in likelihood.py: _DEFAULT_BIN_DELAY_TOL=0.5 (L110),
  _DEFAULT_KERNEL_SUBSAMPLES=2 (L130), _LENS_PARAMS (L193, 7-tuple),
  _amplification_coefficients (L1184) returns (delays,k0,k1,partition) — override
  unpacks `delays,k0,k1,_partition` correctly; _check_candidate_delays (L1592),
  _image_delays (L1362).
- `params` property: MarginalizedExtrinsicLikelihood.params is a class-level LIST
  (L271), so `set(MarginalizedExtrinsicLikelihood.params)` is safe (not a property
  descriptor). Subclass property shadows it correctly.
- Registry probe (import cogwheel.lensing): registered=True; default_likelihood_
  class=LensedMarginalizedExtrinsicLikelihood; standard_params = 19 = 12 intrinsic
  (incl iota,f_ref) + 7 lens (m_lens_msun,z_lens,y1,y2,gamma,beta,kappa). NO
  extrinsic (d_luminosity/ra/dec/psi/phi_ref/t_geocenter absent). Correct.
- __init__ fail-fast ValueError if par_dic_0 missing _LENS_PARAMS; engine rebuilt
  in _set_summary on self.fbin (JSONMixin round-trip safe).
- Candidate-side refusals propagate unswallowed (no try/except in override path).
- Suite: test_lensing_marginalized_likelihood.py: 21 passed (56s).

### Open doc-sync item (STILL OPEN — Librarian at commit)
- INS-5-DOC-1: SPEC.md 'Microlensed sampling layer' + likelihood rows still name
  only LensedIASPrior/LensedRelativeBinningLikelihood; new registered
  LensedMarginalizedExtrinsicIASPrior + LensedMarginalizedExtrinsicLikelihood
  absent. git diff confirms SPEC.md + DATA_CONTRACTS.yaml UNCHANGED. Librarian-
  owned; NOT a code defect. Add SPEC row: coherent-score extrinsic marginalization
  folding total F (F*h data, |F|^2 norm) into reused MarginalizedExtrinsicLikelihood;
  sampled = intrinsic CBC(incl iota) + 7 lens; 'd_luminosity' col = d_app
  (d_L=d_app*sqrt(mu_macro), F009) deferred to post-analysis. No DATA_CONTRACTS
  change (column name unchanged).

### Carried prior doc items (still unconsumed by Librarian)
- INS-4-DOC-1 (LensedIASPrior/LensedPosterior + d_app-deferred note);
  INS-1-001 (SPEC ~line 55 warm lnlike figure + _fid_cache ratio layer).
