---
section: Backlog
---
- [ ] **Build 8f — serving micro-levers: close 4.1x -> ~2x**
  `[→ spec]` — OWNER SCHEDULED (2026-07-20): the post-8b floor ledger
  puts the served lensed likelihood at 6.37 ms median = 4.1x the
  1.56 ms unlensed generic floor (owner target 2-4x; owner wants the
  remaining gap closed as its own build — "let's make it 8f").
  Remaining budget: (a) geometry_partition residual ~2.0 ms — quartic
  image solve + per-image delays + physical kernels + channel switch
  (the caustic search is already Newton, 8b); (b) likelihood-side
  data/norm contraction overhead ~2.3 ms. Both are VALUE-PRESERVING
  optimization targets in the proven 8b-levers mold (HEAD side-by-side
  certification, byte-identity where claimed, F010 falsifications
  through the surviving py_funcs, arc-length-style gauge rules per
  F017). DELIBERATELY NOT bundled into 8d (homogenization): these
  levers need a standing-still baseline to certify against, and 8d's
  baseline moves; they are orthogonal to 8d/8e file-wise.
  ORDERING: after 8e (cusp fast-serving); may overlap the full-box
  training run (training cost is exact-quadrature-dominated, so these
  levers do not materially change it).
  (c) NODE-PARALLEL EXACT EVALUATION (owner-spotted 2026-07-21): the
  positive-parity and saddle wave arms evaluate w nodes in a SERIAL
  loop (~90 ms/node, all nodes independent) — a prange/threaded node
  loop is embarrassingly parallel and value-preserving (8b-levers
  certification mold). Biggest payoff: the full-box TRAINING campaign
  is engine-call-dominated, so node parallelism divides its wall-clock
  by ~core count; also cuts exact-heavy test tiers and any pre-enable
  exact serving.
  PRE-BRIEF DRIVER STEP: profile the 2.0 ms partition residual
  (find_images quartic vs kernels vs switch) so the brief carries
  measured facts, not guesses.
