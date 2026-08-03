# Session: FarFieldChart d/R_c A/B test-spec ruling

Context: EVALUATION build for optional in-memory d/R_c normalization on FarFieldChart
(d_normalized flag on from_engine, per-(gamma,s) rc_table, one divide in serve +
box gate, NO NPZ persistence this build). A/B comparison test authored by Test-Dev.
Honest outcome may be "no benefit."

## Ruling given
1. Endorsed A/B-as-measurement: hard-gate correctness invariants only; RECORD (not
   gate) the near-wall/far-tail eps comparison. The driver's ≥2x near-wall promotion
   gate stays OUT of the test — it's a benefit threshold, not correctness.
   BUT added one hard NON-REGRESSION FLOOR: eps_norm ≤ 1.1·eps_raw in BOTH strata.
   Rationale: a correct smooth positive-scalar reparameterization cannot make the
   envelope materially worse at equal node count; a transposed/off-by-one rc_table
   would. This is the tripwire for silent "normalized worse" bugs that a null-result
   build would otherwise wave through. 1.1 loose enough for noise + honest null.

2. Node-exactness gate on NORMALIZED chart is valid IFF the test queries at nodes of
   the grid the normalized chart actually STORES. Two cases flagged to Test-Dev:
   (a) node locations rescaled too (d_i/R_c) → node maps to node, exactness holds;
   (b) fixed normalized d-grid → physical nodes differ per cell, must query in
   normalized node coords and map back to physical for engine ref, else spurious fail.
   Pure serve-time output divide with same node set → trivially preserved. Test-Dev
   MUST state which the implementation does.

3. (A) list sound, nothing to demote (all exact-arithmetic / discrete-decision
   invariants, provable from the map being a smooth positive bijection — right smell
   test for a hard gate). Additions:
   - Cross-coordinate BOX-GATE CONSISTENCY: pass/reject set identical whether gate
     computed pre- vs post-normalization on shared point set (divide must not move
     the box boundary). Confirm subsumed by train/serve parity or add explicitly.
   - Sharpen min-R_c gate to an ABSOLUTE FLOOR (> floor, not just > 0): R_c near
     machine-eps blows up d/R_c and is a near-cusp the chart is mandated to exclude.

## Note for Dreamer
Recurring pattern in these lensing-chart builds: distinguish BENEFIT thresholds
(never hard-gate on evaluation builds — false RED on honest null) from CORRECTNESS
/ NON-REGRESSION thresholds (hard-gate; provable from the map's properties). The
worse-not-better tripwire (loose ratio ≤ 1.1) is the general trick to catch silent
reparameterization bugs without gating on the benefit itself.
