# Microlensed-GW relative binning: the Chang–Refsdal program

Source: unpublished draft "Relative binning for gravitational-wave microlensing in the
Chang–Refsdal model" (Venumadhav, v5, 2026-07-15; no arXiv ID yet). Manuscript + reference
code shipped in `chang_refsdal_paper_v5_clean.zip` (LaTeX, figures, `code/` with a tested
prototype, benchmark data). This memory is the implementation design manual for adding
microlensed PE to cogwheel.

## Model
Point-mass microlens embedded in the local field of a macro image: external convergence
kappa, shear gamma at orientation beta (Chang–Refsdal). The ENGINE (ec8a276) supports
POSITIVE-PARITY macro images only (1-kappa > |gamma|); the paper does not treat macro
saddles — but the Professor's 2026-07-18 commission DERIVED the saddle treatment (see
"Negative-parity extension" below; full report
`.claude/handoff/lensing/negative_parity_research.md`). Lensed waveform:
h_L(f) = F[w(f), y] h_U(f), with dimensionless frequency w = 8 pi G M_L (1+z_L) f / c^3
(:= xi f — exactly LINEAR in f). Positive parity: source outside the astroid caustic:
2 images; inside: 4.

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
SADDLE analogue (F009-S, verified 2026-07-18): F -> e^{-i pi/2}/sqrt(gamma^2-(1-kappa)^2)
— same magnitude law on |det A| plus a frequency-independent MORSE PHASE e^{-i pi/2};
|F| correction is O(w); phase drifts as w[tau_G + (1/2)ln(w/2) + c0] (tau_G = full Fermat
delay at the macro stationary point A^{-1}y; the w ln w piece is the point-mass core
normalization, present at positive parity too inside C(w)). Pin magnitude AND phase.

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
Built as Build 3f; F008 addendum recorded. CAUTION (rediscovered on the saddle side):
the envelope MUST be built with F demodulated by the same t_min as the relative
carriers — a mismatched convention fakes the beat disease (N 72 vs 24, same config).

## Evaluating F (contour-free, positive parity)
- Point-mass seed: G_PM = C(w) 1F1(1 − iw/2; 1; −iws/2), s=|y|^2, with
  C(w) = exp[pi w/4 + (iw/2)ln(w/2)] Gamma(1 − iw/2).
- Shear via operator identity: G_gamma = exp[(i gamma / 2w) D_beta] G_PM — adaptively
  truncated series; every derivative is another 1F1 (k-ladder with Pochhammer factors).
- kappa via EXACT mass-sheet rescaling: F_{kappa,gamma}(w,y;beta) =
  (1/lambda) exp[iw( ln(lambda)/2 − kappa|y|^2/(2 lambda) )] F_{0, gamma/lambda}(w, y/sqrt(lambda));
  lambda = 1−kappa. Same w. Purely algebraic. HOLDS ON THE SADDLE DOMAIN too for
  lambda > 0 (verified 1e-16); reduces any saddle to pure shear gamma' > 1.
- Image geometry: quartic in u = 1/|x|^2 (general symmetric A matrix), Newton-polished;
  ~190x faster than multistart; special-function evaluation dominates runtime.
- Reference implementation uses mpmath at 60–70 dps — correct ORACLE, far too slow for
  likelihood loops. Production needs a double-precision complex-1F1 kernel
  (series + Kummer transform + k-recurrences + large-|z| asymptotics), numba-compatible;
  stationary-phase regime (w delta >~ 4) needs NO special functions at all.

## Negative-parity extension (Professor research 2026-07-18; report
## .claude/handoff/lensing/negative_parity_research.md — treatment exists)
- Domain split: lambda = 1-kappa > 0 with gamma > lambda = saddle branch (reduced
  gamma' > 1); lambda <= 0 (kappa >= 1, incl. Type III) = NAMED REFUSAL (no clean
  conjugation identity: the log term does not flip with the quadratic).
- Topology (derived by index theorem, verified 4000-source scan, quartic UNMODIFIED):
  sum of signed parities = sign(det A) - 1 = -2; no maxima (tr A = 2 lambda > 0);
  2 images (1,1) both saddles / 4 images (0,1,1,1). Critical curves: the engine's
  astroid formula v(theta) = g c2 +- sqrt(1 - g^2 s2^2) EXTENDS VERBATIM with the
  +- branch, restricted to wedges |sin 2theta'| <= 1/g' around the negative-eigenvalue
  axis: TWO closed lobes; caustics = two 3-CUSP DELTOIDS (An & Evans picture).
- OBSTRUCTION: the shear-operator series has convergence radius EXACTLY the parity
  boundary (Taylor of 1/sqrt(1-g^2) at w->0); measured divergent at ALL w for g'>1
  (best truncation error O(1)). Pade/Borel dead end (target on the Pade cut).
- REPLACEMENT: exact 1D Schwinger representation (any parity, det A != 0):
  F = (w/(2 pi i)) e^{iw|y|^2/2} (pi/Gamma(iw/2)) Int_0^inf dt t^{iw/2-1}
      [(t-iwa/2)(t-iwb/2)]^{-1/2} exp[-w^2 y1^2/(4(t-iwa/2)) - w^2 y2^2/(4(t-iwb/2))],
  endpoint by one integration by parts (continuation in s = iw/2). Branch points at
  t = iwa/2 (lower half-plane for a<0), iwb/2 (upper); det A -> 0 = contour pinch at
  t = 0 (the refusal boundary is manifest). VALIDATED: vs independent 2D rotated-contour
  lens-plane oracle 2.2e-15 (saddle), vs F_op 4e-15 (positive parity), vs point-mass
  closed form exact, mass-sheet identity 1e-16, deep limit F009-S.
- Cancellation law (F001-S): ONE channel, L_S = pi w/4, y-INDEPENDENT (measured
  e^{pi w/4}*1e-16; no 1F1 ladder, no w|y'| exponent, no operator channel). float64
  holds 1e-10 to w~18; DOUBLE-DOUBLE integrand holds 1e-10 to w ~ 64 — same band as the
  existing DD_PRODUCT_CEILING=60. Production = dd Schwinger quadrature, refusal via
  measured paired-rule quadrature error. Steepest-descent alternative: t = (iw/2)v
  extracts e^{-pi w/4} analytically; stationary points of the v-integrand are EXACTLY
  the image quartic roots v* = u = 1/|x|^2 (1D Picard-Lefschetz route, lifts the w
  ceiling; not needed unless w>64 binds).
- Geometric branch and image_kernel (Morse phases, C1/C2) work VERBATIM on saddles
  (measured convergence 2.3e-4 at w*dtau~5). SACR-C carries over with the two-lobe
  nearest-caustic carrier: greedy N = 20-25 over 15 saddle configs incl. fold/cusp
  crossings eta=+-0.002 — the positive-parity band. New risk: tau_c can jump BETWEEN
  lobes across proposals (same class as astroid fold-to-cusp jumps).
- Build shape: two sequential builds (S1 geometry+wave branch; S2 channels/likelihood/
  prior); gates and FINDINGS addenda (F001-S/F005-S/F009-S/F008-note/F004-note) in the
  report Secs. 9-11.

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
The channel/likelihood algebra never references parity — carries to the saddle branch.

## Degeneracies -> sampled coordinates (apply 2207.03508 recipe)
- kappa: EXACTLY unmeasurable (mass-sheet identity above maps it into apparent
  distance + constant time shift + reduced (gamma', y')). Never sample kappa.
- beta: point mass is circular -> only source angle RELATIVE to shear axes matters.
  Never sample beta. Astroid discrete symmetries (quadrant reflections) are FOLDING
  candidates (saddle case: deltoid-pair symmetries, x -> -x between lobes).
- Overall amplitude: sqrt(mu_macro)/d_L -> sample apparent distance d_app; existing
  distance marginalization applies (constant complex scale). Constant lens phase ~
  orbital phase (exact for 22-only; with higher modes — e.g. IMRPhenomXPHM — this
  degeneracy must NOT be assumed, fold/marginalize per mode instead). Saddle macro
  adds a CONSTANT e^{-i pi/2} — degenerate with the same phase freedoms for 22-only,
  NOT with higher modes (a genuine Type-II observable).
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
— an unswallowed exception from a bad proposal must not crash the run. If the saddle
branch is built, the prior becomes two certified domains separated by a refusal band at
the parity boundary.

## Verification obligations (cogwheel value #1)
(i) operator vs mpmath oracle; (ii) reconstruction identity to ~1e-13;
(iii) label continuity across fold/cusp (paper ships these tests — port them);
(iv) multi-component-RB lnL vs brute-force lensed lnL within tolerance;
(v) mass-sheet identity as a likelihood test; (vi) PP/injection recovery.
Saddle branch adds: census/index-sum test, deep-band magnitude AND Morse-phase pins,
independent 2D-oracle anchor, quadrature-error refusal falsification.

Related: `mem:professor/likelihood_and_inference` (relative binning),
`mem:professor/marginalization` (coherent score / time-shift machinery),
`mem:professor/priors_and_coordinates` (sampled-coordinate recipe, folding).
Cited background: 1806.08792, 2404.02435, 2207.03508; An & Evans 2006 (Chang-Refsdal
revisited, gamma>1 deltoid caustics — consistent with our numerics).
