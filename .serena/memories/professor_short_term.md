# Professor Short-Term — Session 2026-07-18 (envelope research commission)

## Mission and outcome
RESEARCH mode: find a beat-free decomposition of the Chang-Refsdal
transition-band kernels (Build-3d disease: kernels bind at 50-90 nodes).
**SOLVED** — full deliverable in `.claude/handoff/lensing/envelope_research.md`.

## The decomposition (SACR-C, certified)
F(w) = sum_a e^{iw tau_a} S_a(w) H_a(w) + e^{iw tau_c} E(w).
- H_a = geometry.image_kernel (closed form), tau_c = nearest_caustic_point
  carrier delay (the engine's virtual_delay).
- S_a = smootherstep(w * |tau_a - tau_c|, 0.5, 4.0) — criticality separation,
  NOT the F008 nearest-neighbour min. Key theorem: switch scale ==
  demodulation distance, so all O(1) content in E has phase <= 4 rad.
- E = e^{-iw tau_c}(F - sum S_a H_a e^{iw tau_a}) is the ONLY interpolated
  object (cubic spline in ln w, Re/Im).

## Certified numbers (scripts envelope_exp1..6.py in session scratchpad)
- Identity: recon == exact_total at 2e-16 .. 3e-16 rel.
- 2-decade windows: greedy N = 19-26 over 25 configs (5 anchors + 8 fold/cusp
  crossings eta=+-0.002/0.01 both sides + 12 random) — config-independent.
  Full 2.7-4.6-decade bands: N = 20-42.
- Control same-oracle: current engine kernels need N = 40-53.
- Production placement: greedy transplant across configs FAILS; LOO adaptive
  (seed 8 log, split worst flanks, stop LOO<4e-3) gives N = 30-44,
  self-certifying, all eps < 1e-3 (calibrated: 8e-3 stop starts failing).
- Cost: F_op_grid 0.41 ms/node batched → 12-18 ms/eval (LOO), 8-11 ms oracle
  bound; vs 20-37 ms current. 10 ms gate needs ratio layer or tuned seeding.
- max|S_a H_a| <= 1.30 incl. crossings (F008 intent preserved: merging images
  have tau_a -> tau_c so the gate is MORE conservative; accidental
  degeneracies no longer stall — crown fixed).
- Floor sensitivity: N unchanged at floors 0.15/0.05/0.01 on anchor windows.

## Paper prototype clarification (important correction of lore)
The 6-11 node claim = greedy nodes on candidate/fiducial RATIOS q_a over
w in [5,40] (0.9 decades), floor 0.15 max|F|. Per decade (~7-12) it MATCHES
our kernel-level result (~9-12/decade). The prototype's partition is
block-structured (persistent analytic + cluster-local residual), never the
engine's flat 1/4-weighted full-F split — that flat split + uniform residual
weights is the root cause of the beat disease (verified in _gauge.py).

## Dead ends (do not retry)
- Parametric 1/w^3+1/w^4 tail fits from coarse nodes: unnecessary + biased
  (near_cusp fit blew up to eps=1.0).
- Node-position transplant across configs: fails at 1e-1.
- Per-image wave residual R_j (Build 3e): still nonexistent, NOT needed.

## Dreamer consolidation notes
- professor/microlensing_chang_refsdal updated this session with the durable
  decomposition + bounded-phase theorem.
- If a build lands this: FINDINGS needs an addendum superseding the F008
  switch-separation rule (full-cluster min -> |tau_a - tau_c|) in the
  channel-construction context (branch gate _min_delay_separation untouched).
