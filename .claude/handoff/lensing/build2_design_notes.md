# Build 2 design notes: multi-component RB likelihood (math locked with the user)

Derived and agreed in the 2026-07-16 design thread. These are REQUIREMENTS-level
notes for the Build-2 brief; the Architect owns the plan.

## Objects
Candidate lensed waveform: h(f) = F(f) h_U(f), F = sum_a e^{i w(f) tau_a} K_a(w),
h_U = sum_m h_m (modes). Fiducial: subscript 0. Per-component ratios FACTOR:
  r_{am}(f) = rho_a(f) q_m(f),
  rho_a = e^{2 pi i f dt_a} kappa_a,  kappa_a = K_a(w(f))/K_a0(w0(f))  [slow],
  q_m = h_m/h_m0  [slow],
  dt_a = (xi tau_a - xi0 tau_a0)/(2 pi)  [EXACTLY constant in f since w = xi f].

## The trap (user probed this; the answer is locked)
NEVER product-of-summaries: sum_f X(f)Y(f) != (sum X)(sum Y). The factorization
is RAPID-vs-SLOW, not F-vs-h_U. Rapid x rapid stays jointly inside the f-sum.

## Norm-term summaries (delay-continuous)
  T^(p)_{mn,b}(dt) = 4 sum_{f in b} e^{2 pi i f dt} h_m0 h_n0^* / S_n (f-f_b)^p df
Contains NO image index; carrier and mode-pair product multiplied at full
resolution inside the sum, exactly. Image pairs enter by EVALUATION at their
delays. The slow fiducial product K_a0 K_c0^*(f) is Taylor-expanded within bins:
  B^(p)_{(am)(cn),b} = [K_a0 K_c0^*](f_b) T^(p)_{mn,b}(Dt_ac,0)
                     + [K_a0 K_c0^*]'(f_b) T^(p+1)_{mn,b}(Dt_ac,0) + O(Df_bin^2 K'').
=> norm term needs moments p <= 3 (standard 0..2, +1 for the fiducial-K slope).
Data-term analog A^(p)_{m,b}(dt) = 4 sum_{f in b} d^* e^{2 pi i f dt} h_m0/S_n (f-f_b)^p
== the coherent-score z_m(t) timeseries structure. FFTs in SETUP ONLY.

## Sequential contraction (additive, hot-path FFT-free)
Carrier-extract: T^(p)(dt) = e^{2 pi i f_b dt} Ttilde^(p)(dt); the envelope
Ttilde varies on scale 1/Df_bin (~20 ms at 50 Hz bins) => a few dt-grid nodes
span the physical delay range.
Stage A (modes, no image index): U_b(dt_k) = sum_{mn} q_m q_n^* T_{mn,b}(dt_k)
  -> M^2 x N_grid x N_bins  (~25 x 6 x 100 ~ 1.5e4).
Stage B (images, no mode index): evaluate U_b envelope at the ~10 candidate
pair delays Dt_ac (1-D interpolation, analytic carrier), weight by
rho_a rho_c^* Kbar_ac -> n_img^2 x N_bins (~1e3).
Total ~2e4 flops — same order as today's HM contraction; scaling M^2 + n_img^2,
NOT M^2 n_img^2. (User flagged the naive multiplicative version is ~1e5-1e6
flops, dangerously close to a brute-force overlap sum — unacceptable.)

## dt vs Dt handling
- Fiducial absolute delays Dt_a0: inside summaries at full resolution — exact.
- Candidate residuals dt_a: v1 = plain linear RB (linearization captures the
  bin-center carrier automatically; per-bin error ~ (pi Df_bin dt)^2), valid
  |dt| <~ 1-3 ms for 30-100 Hz bins — covers the posterior, not the full prior.
- COMMON shift (degenerate with t_c): extract via the BaseLinearFree idiom
  (cogwheel already time-aligns the (2,2) to the reference and returns the
  shift); only DIFFERENTIAL residuals ddt_ac remain in binned ratios.
- General case (prior-wide): evaluate T, A at CANDIDATE delays (delay-continuous)
  — whole delay phase exact, only kappa_a kappa_c^* q_m q_n^* binned; removes the
  small-dt restriction. This is the principled upgrade path; shares its skeleton
  with the coherent-score machinery.
- GUARDS (hard requirements): assert pi * Df_bin * dt_max < tol at setup;
  lens-aware bin selection (add lens dephasing to the bin criterion);
  brute-force lnL agreement test is the arbiter.

## Singled-out unit test
The fiducial-K within-bin Taylor is the one NEW approximation: test it where
K_a0 varies fastest (near the switch region w*delta_j ~ rho_0). If it fails
there, fall back to exact B tensors for the offending bins only (hybrid).

## Other locked requirements
- Timing assertions in tests: contraction subdominant to the coarse-node
  waveform call; K_a evaluation subdominant/comparable to it.
- v1 may be 22-only (M=1: 10 pairs x 3 moments — trivial); the sequential
  contraction becomes MANDATORY when HM lands.
- LensedWaveformGenerator: composition (wrap a WaveformGenerator), multiply
  each mode by F; F is detector-independent (source-side) so existing
  extrinsic handling survives.
- Normalization convention: F includes sqrt(mu_macro) (F(w->0) =
  1/sqrt((1-kappa)^2-gamma^2), NOT 1) — do not double count against distance.
  Overall amplitude belongs to apparent distance d_app (Build 3 samples it).
- Brute-force reference: lnL via direct F(w(f))*h_U(f) at full resolution
  (channels.amplification()) — the agreement test tolerance is the build gate.
