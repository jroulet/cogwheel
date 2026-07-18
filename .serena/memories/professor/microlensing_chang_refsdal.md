# Microlensed-GW relative binning: the Chang–Refsdal program

Source: unpublished draft "Relative binning for gravitational-wave microlensing in the
Chang–Refsdal model" (Venumadhav, v5, 2026-07-15; no arXiv ID yet). Manuscript + reference
code shipped in `chang_refsdal_paper_v5_clean.zip` (LaTeX, figures, `code/` with a tested
prototype, benchmark data). This memory is the implementation design manual for adding
microlensed PE to cogwheel.

## Model
Point-mass microlens embedded in the local field of a macro image: external convergence
kappa, shear gamma at orientation beta (Chang–Refsdal). POSITIVE-PARITY macro images only
(1-kappa > |gamma|); macro saddles (Type II — common in strong-lensing pairs!) are OUT OF
SCOPE of the current formalism — record as a limitation. Lensed waveform:
h_L(f) = F[w(f), y] h_U(f), with dimensionless frequency w = 8 pi G M_L (1+z_L) f / c^3
(:= xi f — exactly LINEAR in f). Source outside the astroid caustic: 2 images; inside: 4.

## The w -> 0 macro limit (established 2026-07-17, F009; supersedes an earlier FALSIFIED
## Professor ruling that called this a gamma/(2w) engine singularity — it is not)
F(w) -> 1/sqrt((1-kappa)^2 - gamma^2) = sqrt(mu_macro) as w -> 0: the EXACT macro-image
geometric-optics magnification (mu_macro = 1/det(A) = 1/[(1-kappa)^2-gamma^2]), NOT 1 and
NOT a numerical artifact. Confirmed flat (frequency- and mass-independent) across many
decades of w, matching the closed form to ~1e-8 relative. Consequence: any "unlensed
limit" fixture/test that expects F->1 must use gamma=kappa=0 (genuinely trivial macro
lens) — a sheared/converged lens's amplitude never relaxes to 1 no matter how small w
gets. Do NOT "fix" the flat offset with a small-w short-circuit forcing F->1+O(w); that
would inject a real discontinuity and destroy the exact macro limit. This value multiplies
the whole waveform, so it composes with the apparent-distance sampling below (mu_macro
folds into d_app, not into F itself, once F is defined against the true unlensed h_U).

## The decomposition (the paper's core)
Direct RB fails because interference fringes move with lens parameters (ratio needs ~all
frequencies: 70–135 nodes). Fix: F = sum_{a=1..4} e^{i w tau_a} K_a(w) — keep image-delay
phases ANALYTIC, interpolate only the slow K_a. Construction:
- Stationary-phase kernels H_a = sqrt|mu_a| e^{-i pi n_a/2}[1 + iC1/w + C2/w^2] with
  explicit C1, C2 polynomials (paper Appendix B) — the resolved-image targets.
- EXACT residual projection: K_a = K_hat_a + alpha_a e^{-i w tau_a} R(w) with
  R = F_op − sum_b e^{i w tau_b} K_hat_b. The coherent sum equals the full wave-optics
  F_op at EVERY frequency (machine precision ~1e-15); interpolation is the only error.
- The prototype's partition is BLOCK-structured (chang_refsdal_topology_stable.py):
  persistent images always carry analytic H under their own carriers; only the CLUSTER
  residual, demodulated at the critical-point delay, is split cluster-locally. The
  paper's 6–11-node claim = greedy-adaptive nodes interpolating candidate/fiducial
  RATIOS q_a over w in [5,40] (0.9 decades), floor 0.15 max|F| — i.e. ~7–12 nodes per
  decade, NOT a multi-decade raw-kernel claim.
- Label continuation is path-based (assignment problem on lens-plane markers);
  far-away proposals need a reset convention or short path from fiducial. The SUM is
  label-permutation invariant; only ratio smoothness needs consistent labels.

## Beat-free transition envelope (Professor research, 2026-07-18; full derivation +
## certification in .claude/handoff/lensing/envelope_research.md)
The engine's flat 4-label gauge (each unresolved channel = (1/4) e^{-iw tau_a} F, uniform
residual weights) puts the FULL total, with all carriers, into every unswitched kernel —
the root cause of the Build-3d beat disease (50–90 nodes). The F008 nearest-neighbour
switch separation additionally stalls on ACCIDENTAL delay degeneracies (crown: two
near-degenerate delay pairs from quasi-symmetry, images NOT merging in the lens plane).
The certified fix (SACR-C):

    F(w) = sum_a e^{iw tau_a} S_a(w) H_a(w) + e^{iw tau_c} E(w),
    S_a = smootherstep(w |tau_a - tau_c|, 0.5, 4),  tau_c = critical/virtual delay,
    E   = e^{-iw tau_c} (F - sum_a S_a H_a e^{iw tau_a})  — the ONE interpolated object.

Bounded-phase theorem: the switch scale IS the demodulation distance, so any O(1)
content in E carries demodulated phase <= rho_END = 4 rad; switched channels contribute
only the O(w^-3) saddle-asymptote tail (bounded visible cycles). Merging images have
tau_a -> tau_c, so the gate is MORE conservative than F008's full-cluster min (measured
max|S_a H_a| <= 1.3 incl. eta=+-0.002 crossings); accidental degeneracies no longer
stall (and when degenerate WITH tau_c they are harmless: small carrier separation = no
beat). Deep-unresolved limit carries F009 verbatim. Certified: identity ~2e-16;
2-decade windows: greedy N = 19–26 over 25 configs (config-independent, ~9–12/decade);
full 2.7–4.6-decade bands N = 20–42; control (current kernels, same oracle) N = 40–53.
Production node placement: position transplant across configs FAILS; use LOO adaptive
refinement (stop max LOO < 4e-3): N = 30–44, self-certifying. Cost 0.41 ms/node batched.
Dead ends: parametric 1/w^3 tail fits from coarse nodes (biased, blew up at near-cusp);
per-image wave residual R_j (Build-3e premise) — nonexistent and NOT needed.
If built: supersedes the F008 switch-separation rule in channel construction (branch-gate
_min_delay_separation untouched) — FINDINGS addendum required.

## Evaluating F (contour-free)
- Point-mass seed: G_PM = C(w) 1F1(1 − iw/2; 1; −iws/2), s=|y|^2, with
  C(w) = exp[pi w/4 + (iw/2)ln(w/2)] Gamma(1 − iw/2).
- Shear via operator identity: G_gamma = exp[(i gamma / 2w) D_beta] G_PM — adaptively
  truncated series; every derivative is another 1F1 (k-ladder with Pochhammer factors).
- kappa via EXACT mass-sheet rescaling: F_{kappa,gamma}(w,y;beta) =
  (1/lambda) exp[iw( ln(lambda)/2 − kappa|y|^2/(2 lambda) )] F_{0, gamma/lambda}(w, y/sqrt(lambda));
  lambda = 1−kappa. Same w. Purely algebraic.
- Image geometry: quartic in u = 1/|x|^2 (general symmetric A matrix), Newton-polished;
  ~190x faster than multistart; special-function evaluation dominates runtime.
- Reference implementation uses mpmath at 60–70 dps — correct ORACLE, far too slow for
  likelihood loops. Production needs a double-precision complex-1F1 kernel
  (series + Kummer transform + k-recurrences + large-|z| asymptotics), numba-compatible;
  stationary-phase regime (w delta >~ 4) needs NO special functions at all.

## Multi-component RB likelihood
h_L = sum_a r_a h_{a0}; r_a = (h_U/h_U0) e^{i[w tau_a − w0 tau_a0]} K_a/K_a0.
Summaries: A^{(0,1)}_{a,b} (data-component), B^{(0,1,2)}_{ac,b} (pair; 10 pairs for
4 labels). KEY: since w ∝ f, w tau_a − w0 tau_a0 = 2 pi f dt_a with CONSTANT dt_a
(dt = xi tau/(2 pi) = 4 G M_L(1+z) tau / c^3 ≈ 2e-5 s (M_L/Msun)(1+z) tau) — image
delays are pure TIME SHIFTS. Shifted summaries are short DFTs, tabulated on a delay
grid / FFT — mathematically identical to cogwheel's coherent-score z(t) timeseries
machinery. With higher modes: components (a, m) with r_{am} = rho_a q_m (factored
ratios — F multiplies the whole waveform); pair summaries collapse to mode-pair
summary FUNCTIONS of a continuous time shift; images never add a stored tensor index.
With the SACR-C envelope form, a 5th channel at carrier tau_c (15 pair summaries) is
the clean shape: the four analytic channels are closed-form (no engine nodes at all).

## Degeneracies -> sampled coordinates (apply 2207.03508 recipe)
- kappa: EXACTLY unmeasurable (mass-sheet identity above maps it into apparent
  distance + constant time shift + reduced (gamma', y')). Never sample kappa.
- beta: point mass is circular -> only source angle RELATIVE to shear axes matters.
  Never sample beta. Astroid discrete symmetries (quadrant reflections) are FOLDING
  candidates.
- Overall amplitude: sqrt(mu_macro)/d_L -> sample apparent distance d_app; existing
  distance marginalization applies (constant complex scale). Constant lens phase ~
  orbital phase (exact for 22-only; with higher modes — e.g. IMRPhenomXPHM — this
  degeneracy must NOT be assumed, fold/marginalize per mode instead).
- Overall time: min image delay degenerate with t_c (subtract-min convention).
- Well-measured lens observables: fringe SPACING = dimensionful delay differences
  dt_ac (sample ~ln dt of dominant pair, NOT raw M_L), fringe CONTRAST = |K_a/K_c|
  flux ratios. Near folds the signed caustic distance eta_s is likely the
  well-measured coordinate. Net: 6 raw lens params -> ~4 sampled ones
  {ln dt, contrast, folded source angle, gamma'}.
- Research direction (v2): in the resolved limit the lens acts as
  h_L = sum_a c_a h_U(t − dt_a) — same structure the coherent score marginalizes
  (amplitudes + time shifts) -> lens-sector importance-sampling marginalization
  using the existing z(t) machinery is plausible.

## Build-3/4 sampler requirement (flagged 2026-07-17, not yet implemented)
Macro saddles (1-kappa <= |gamma|) raise `geometry.LensDomainError` at construction and
symmetrically in both `lnlike`/`lnlike_bruteforce`; `F_op` raises `CancellationError` near
the certified-domain edge (gamma_eff ~0.5). A sampler must bound the prior to the
positive-parity/certified region, or map these refusals to lnL=-inf at the proposal level
— an unswallowed exception from a bad proposal must not crash the run.

## Verification obligations (cogwheel value #1)
(i) operator vs mpmath oracle; (ii) reconstruction identity to ~1e-13;
(iii) label continuity across fold/cusp (paper ships these tests — port them);
(iv) multi-component-RB lnL vs brute-force lensed lnL within tolerance;
(v) mass-sheet identity as a likelihood test; (vi) PP/injection recovery.

Related: `mem:professor/likelihood_and_inference` (relative binning),
`mem:professor/marginalization` (coherent score / time-shift machinery),
`mem:professor/priors_and_coordinates` (sampled-coordinate recipe, folding).
Cited background: 1806.08792, 2404.02435, 2207.03508.
