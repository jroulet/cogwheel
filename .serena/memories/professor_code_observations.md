# Professor — code observations (personal, does NOT propagate)

cogwheel code-level details the Professor accumulates (function behavior, call
order, data layouts, numerical gotchas). Personal to this checkout — soft-blacklisted.

- `channels.py` `_channel_switch`/`_min_delay_separation`: the neighbor-
  distance comparison must run over ALL channel labels (incl. virtual/
  parked), not real-only — real-only was the crown-blowup bug, fixed at
  F008 (see FINDINGS.md). `_min_delay_separation` uses the same
  real-only pattern for the wave/geometric branch gate; audited as not
  the crown cause (geometric branch = real saddles only) but worth
  re-checking against the paper's cluster-separation definition if
  issues surface there later.
- Zero-noise F->1 floor in `likelihood.py`: after anchoring on a
  gamma=kappa=0 candidate (genuinely F->1), the residual RB gap traces
  to a construction asymmetry — `_set_summary` builds `_h0_edges` with
  `disable_precession=False` forced plus `_stall_ringdown`, while
  `_candidate_bin_ratios` builds candidate edges with neither, so the
  ratio != 1 in the stalled-ringdown band even at F=1 (F007). Fix
  direction: make `_candidate_bin_ratios` mirror `_set_summary`'s edge
  build (apply the same fiducial fadeout/f_99 + forced precession to
  candidate edges) rather than computing a full-res candidate strain
  per eval.
- `F_op` (`operator.py`) CancellationError fires near gamma_eff~0.5 (the
  order-42 shear-series tail crosses the 1e-10 refusal cut there) — this
  is the marginal certified-domain edge for macro-saddle-adjacent
  configs, not a bug; don't widen the refusal threshold to route around it.
- Batched contraction (`F_op_grid`/`_grid_certified`, Build 3c): scalar
  `F_op` delegates to the batched core, so scalar suites auto-exercise
  it. Certified rel-err vs the mpmath oracle grows exponentially with
  cancellation exponent L (~1e-16 at L~0 -> ~1e-10 at L~44), then the
  F005 refusal cuts in — the 1e-10 accuracy ceiling coincides with the
  refusal boundary; certify-XOR-refuse held with zero solo-vs-batch
  decision flips (no cross-node convergence-state leakage).
