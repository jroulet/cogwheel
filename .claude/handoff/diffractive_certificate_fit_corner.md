# Build: fix w_low_fit high-gamma/small-r corner + move the full bake to the driver

## Mission

The `diffractive_certificate_fit_fix_aliasing` build achieved the aliasing
fix (off-grid never-over-serve now holds, 224/224 conservative) but WEDGED on
the long in-build bake (a ~32 min `--scale full` run exceeds the SDK's 1200s
inter-message ceiling) and the completed bake revealed a FIT-QUALITY problem:
a localized 2.06x raw over-prediction at gamma~0.41, r~0.55 forces the
de-rate down to 0.485, crushing tightness to median 0.49 (plan wanted >= 0.7).

This build fixes the fit at that corner so the de-rate can rise toward the
0.85 hard floor, AND moves the full calibration bake OUT of the build into a
driver step (the bake is a bulk sweep; in-build it cannot fit the SDK timeout).

## Measured facts (completed bake at SHA 068a2b1, engine-verified)

- New 4-harmonic / 32-theta surface: off-grid 224/224 conservative (never
  over-serve, worst ratio exactly 1.0 at the de-rated surface) — the INS-4-001
  guarantee is RESOLVED.
- Tightness FAIL: grid 318/908 (35%), off-grid 100/224 (45%) above 0.5x true;
  median grid 0.487 / off-grid 0.492 (need >= 0.7).
- Root cause of the tightness crush: `de-rate = min(0.85, 1/max_overpred) =
  0.4846` because the RAW (un-de-rated) surface over-predicts by 2.06x at a
  LOCALIZED corner: gamma=0.41, r=0.55, theta in {2.454, 5.596}
  (raw fit 7.1 vs true 3.5). This is the gamma 0.4-0.5 band near the
  positive-parity wall — where the ceiling collapses steeply and the order-8
  certificate used to refuse. The degree-2 log-log poly cannot follow the
  steep collapse there, so the de-rate absorbs it globally.
- The bake is ~32 min (972 grid + 240 off-grid + 12 random rows x ~1.75 s)
  and CANNOT run in-build (SDK 1200s ceiling; the wedge is why this build
  exists).

## Scope

IN:
1. FIT CORNER FIX (the 2.06x over-prediction at gamma 0.4-0.5, r~0.5):
   - The representation must follow the steep ceiling collapse toward the
     positive-parity wall. Options the build should evaluate and pick by
     measurement:
     (a) higher-degree polynomial (degree 3) in the log-features;
     (b) a REGION SPLIT at gamma ~ 0.35 (two polynomial surfaces: the
         low-gamma smooth band and the high-gamma steep-collapse band),
         with continuity at the seam;
     (c) extra basis terms capturing the wall collapse (e.g. a term in
         log(1-gamma') interaction with r).
   - WHATEVER is chosen, the acceptance is: de-rate >= 0.70 (up from 0.485),
     off-grid tightness >= 0.5x on >= 80% (up from 45%), median >= 0.6, AND
     the never-over-serve guarantee still holds off-grid (exact float64,
     conservative 100%). If the corner genuinely cannot be modeled without
     breaking conservativeness, fall back to a REGION SPLIT with a
     conservative high-gamma surface and document the achieved de-rate.
   - The fit script gains the new representation; the shipped
     `w_low_fit`/`_fit_features`/`_fit_model` in `_diffractive.py` must match
     it exactly (mirror-fidelity: the production code and the bake script
     share the basis builder; never two copies).
2. BAKE = DRIVER STEP: the build must NOT run `--scale full` (or anything
   > ~5 min). The build runs a SMOKE-scale bake only (seconds, proves the
   pipeline end-to-end) and commits the mechanism. The FULL bake and the
   final coefficient paste is a POST-BUILD DRIVER step (the driver runs
   `scripts/fit_diffractive_certificate.py --scale full`, pastes the emission
   verbatim, and re-validates). The smoke bake's emission may be committed as
   a placeholder IF clearly marked provisional; the acceptance numbers below
   are for the driver-finalized bake.

OUT (do not touch):
- The served series `diffractive_amplification` (exact order-16).
- The aliasing fix already committed (n_harm=4, 32-theta grid, off-grid
  de-rate mechanism) — keep it.
- Rung S / macro-saddle engine-host, refusal-gate semantics.
- Any surrogate-chart or campaign work.

## Acceptance (for the build's portion; final bake is driver-verified)

- The new representation is SHIPPED and the fit script + `w_low_fit` share
  it (one basis builder). A smoke-scale bake runs in-build (< 2 min) and
  proves the pipeline; the emission it produces is committed as a provisional
  placeholder with a clear `# PROVISIONAL` marker.
- The smoke bake's corner behavior is SANE: at gamma in {0.40, 0.45, 0.50} x
  r in {0.5, 0.55} the raw surface over-prediction is < 1.5x (down from
  2.06x) — demonstrating the corner fix is real, not just re-de-rated.
- Tests: the off-grid never-over-serve pin stays green; the tightness bar is
  updated to the achievable target the build's measurement supports
  (state the achieved de-rate + median in the completion record). The
  self-falsification teeth stay (derate=1.0 trips over-serve).
- NO long bake in-build. If the Coder tries, the build must reject it.

## Constraints

- Branch `claude-dev`. Slow tiers stay gated. In-build tests FAST; the ONLY
  in-build engine work is the smoke bake (< 2 min).
- Spec/TODO workflow: `[→ spec]`.
- The driver step (full bake + paste + re-validate) is the SDG post-build
  convention: bulk sweeps NEVER run in a build. The completion record must
  state that the shipped coefficients are PROVISIONAL until the driver bakes.
