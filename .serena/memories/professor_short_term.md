# Professor Short-Term — Session 2026-07-18 (Build 3d consultation)

## Context
Consulted on the h_L = F*h_UL factorization speedup for `LensedRelativeBinningLikelihood.lnlike`.
Target: 41 ms -> <=10 ms by reducing engine node count from ~100 to ~10-12.

## Key Findings

1. **The correct factorization is NOT "interpolate delay-removed per-channel
   amplitudes"** — those (artificial_a) still oscillate at beat frequencies
   tau_a - tau_cluster. The correct object to interpolate is `exact_total(w)` on
   the WAVE SEGMENT ONLY, with the geometric segment evaluated analytically at
   dense resolution (geometric_amplification is cheap: no 1F1).

2. **Node count is Nyquist-limited by delay-spread oscillations in exact_total.**
   For 4-image crown: ~15-18 nodes on the wave segment. For 2-image well-separated:
   ~6-10. For near-fold 2-image (tiny delta_min => wide wave band): 80-120 nodes —
   lever A does NOT solve this regime.

3. **Branch transition at w_branch = max(4/delta_min, 48/|y'|)** is a hard C2 kink
   in exact_total. Must segment there. The geometric segment above w_branch is FREE
   (no engine nodes needed).

4. **Predicted performance with lever A:**
   - Crown 4-image: ~9-11 ms (marginal on 10 ms gate, meets 12 ms relaxed gate)
   - 2-image well-separated: ~6-8 ms (comfortable)
   - Near-fold: ~30-45 ms (needs lever B in future build)

5. **Lever B (3D surrogate table) deferred to Build 4.** Ship lever A alone.

6. **The paper's 6-11 node claim** was measured in the fully-resolved regime where
   w_min > w_branch for the test configs. In that regime K_a ~ target_a + O(1/w^3)
   residual, which IS smooth. The transition band does not enjoy this.

## Dreamer Consolidation Notes
- Update `professor/microlensing_chang_refsdal` with the factorization analysis
  (what is/isn't slowly varying, Nyquist constraint, branch segmentation).
- The near-fold regime (delta_min < 0.1) is flagged as the residual hard case for
  lever B scoping.
- The claim "delay-removed amplitudes are smooth" in the existing memory is
  imprecise — should be qualified: smooth only in the resolved regime.