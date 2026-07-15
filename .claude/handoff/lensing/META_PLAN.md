# Microlensed-PE program — meta-plan and live status

Driver: the session agent (meta-planning only; each build's Architect owns its
internal plan). User directive 2026-07-16: drive autonomously to the finish line,
approve/reject plans via the file gate, give feedback, no further user input.

## Source materials
- Paper + tested prototype: `.claude/spec/lensing_paper/` (tex, pdf, code/, data/).
  UNPUBLISHED — stays in .claude/ (excluded from main sync).
- Professor topic memories: `professor/microlensing_chang_refsdal` (design manual),
  `professor/likelihood_and_inference`, `professor/marginalization`,
  `professor/priors_and_coordinates` (coordinate recipe + folding).

## Design decisions locked in with the user (2026-07-16 thread)
1. Multi-component RB with image-delay phases analytic; K_a interpolated.
2. Summary structure: NEVER product-of-summaries. Rapid×rapid stays inside the
   f-sum as delay-continuous summaries T^(p)_mn,b(δt) (and data-side A^(p)_m,b(δt)
   ≅ z_m(t) timeseries); slow fiducial K_a0*K_c0 Taylor-expanded within bins
   (costs one extra moment: p ≤ 3 for the norm term).
3. Hot path: NO FFTs (setup only). Sequential contraction — modes first
   (M² × few δt-grid nodes × bins), then images via envelope interpolation at
   10 pair delays (n_img² × bins). Additive M²+n_img², NOT multiplicative.
   Contraction must stay subdominant to the coarse-node waveform call.
4. δt vs Δt: fiducial absolute delays exact inside summaries; candidate residual
   δt handled by (v1) linear RB + lens-aware bin criterion + guard assert
   [π Δf_bin δt_max < tol], (general) delay-continuous evaluation at candidate
   delays. Common time shift via the BaseLinearFree idiom.
5. Degeneracies: kappa NEVER sampled (exact mass-sheet identity); beta NEVER
   sampled (circular point mass); sample reduced (gamma', y-in-shear-frame);
   overall amplitude -> apparent distance d_app (existing distance machinery);
   min-delay convention -> t_c. Astroid quadrant symmetries -> folding candidates.
6. Special functions: mpmath is ORACLE ONLY (tests). Production: double-precision
   complex-1F1 kernel (series + Kummer + k-ladder recurrences + large-|z| asymptotics),
   numba-compatible; stationary-phase regime needs no special functions.
7. Verification obligations (every build): tolerance tests vs exact/brute-force
   references (cogwheel value #1); timing assertions so contraction/K_a-eval
   regressions fail tests.

## Build sequence and status
- [LAUNCHED] Build 1 — lens engine: cogwheel/lensing/chang_refsdal/
  (geometry, operator, channels) + oracle/reconstruction/continuity tests.
  Brief: build1_brief.md. No likelihood integration.
- [PENDING] Build 2 — LensedWaveformGenerator + multi-component RB likelihood
  (decisions 2-4 above) + brute-force lnL agreement tests. Brief written after
  Build-1 review (API of channels feeds in).
- [PENDING] Build 3 — priors/sampled coordinates + folding + injection-recovery
  validation. Brief after Build-2 review.
- Optional research spike (post-Build-3, judgement call): lens-sector
  importance-sampling marginalization on the z(t) machinery.

## Driver protocol (for future context windows)
1. Launch: `.claude/sdk/launch_build.sh lensing_buildN .claude/handoff/lensing/buildN_brief.md`
   (file-gate approval; watchdog attaches; Monitor pattern printed in log header).
2. On plan-ready: read /tmp/lensing_buildN_approval/plan.json; REVIEW against this
   file's locked decisions; approve (touch plan_approved) or reject with concrete
   feedback (write plan_rejected). Respond promptly (watchdog clock runs).
3. On build end: review commit + report; run the build's tests myself; update this
   file's status; write next brief; launch next build.
4. If a build FAILS: diagnose from log; fix forward via a follow-up brief (or
   direct fix for infra-level issues only — physics/code goes through builds).
5. Between builds: /doc-sync if librarian backlog nears the hard block (>5).
