# Inspector Short-Term Observations

## 2026-08-17 review — saddle band-split serving revival (WP1/WP2/WP3)

Scope: reviewed all uncommitted changes. Code files changed:
`cogwheel/lensing/likelihood.py` (+~406) and
`cogwheel/tests/test_lensing_saddle_serve_gate.py` (+~1180). (Also
`.claude/agent_state/*`, `.serena/memories/*` — non-code.) SPEC.md was in the
plan but NOT changed.

VERDICT: PASS. No new bug/crash beyond HEAD. Tests exemplary (derived,
independent-oracle, self-falsifying, non-vacuous).

### What was verified
- WP1 `_band_split_mask(dense_w, split)` — factored the Born inline band-split
  logic into a module-level helper. Confirmed BYTE-IDENTICAL to prior inline
  `below_mask=((dense_w<=w_trust) if band_split else np.ones(...))`; Born's own
  `eff_ceiling` w_trust-nulling guard stays. `band_split = split is not None and
  w_lo < split < w_hi` (strict interior); below_mask uses inclusive `<=`.
- WP2 `_saddle_farfield_analytic` — whole-band admit fast path is byte-identical
  to HEAD (zeros envelope, NO engine call). Band-split else-branch: refuse if
  min_sep None/<floor, refuse if w_split None or >=w_hi or >150; else engine
  populates below-split via `_engine_envelope_below_split`, rung zeros above.
  `_saddle_c3_split_point` uses exact C/w^3 cube-root inversion at w_ref=1.0;
  None-ness is w-independent (verified consistent with the gate).
  `_saddle_min_image_sep` shared by gate + rung (single source).
- WP3 `_ppgo_above_ceiling` — gate changed from `w_lo*min_delta_tau<RHO_END` to
  `W_CEILING_SCHWINGER_QD*min_delta_tau<RHO_END → None`. Split at 150; engine
  below via `_engine_envelope_below_split`, fold envelope zeroed below.

### RESOLVED candidate (do not re-flag)
- WP2 `_engine_envelope_below_split` pads a size-1 sub-band with
  `pad_w=float(dense_w.max())` which can be w_hi>150. Initially flagged as a
  possible SchwingerCertificationError absent a WP2 resolution gate. RESOLVED:
  on the SAME draw HEAD's fall-through seed engine path also evaluates the exact
  engine at w_max>150 (`seed_w` max IS w_max, pad_w=w_max), so any raise is
  byte-identical terminal behavior to HEAD; if w_hi is resolved, evaluate routes
  to geometric (no raise) and band-split serves correctly. Not a regression.
- INS-1-001 (double-mask crash `np.asarray(geom.images)[real_mask]` on an
  already-real-only array → IndexError) is CONFIRMED FIXED at both census and
  serve-rung sites (`real_images=np.asarray(geom.images)` directly). This is the
  same DOUBLE-MASK pattern recorded in coder_knowledge 2026-08-14.

### Non-blocking carry-forwards
- WP3 whole-band-above coverage narrowing: when the ceiling split does not fall
  strictly interior, admission narrows conservatively (refuse-only "deferred 2b
  residual", documented). Safe direction (loses coverage, never over-serves).
  Design/trivial, accepted — NOT a defect.
- Leaky-gate witness `_C3_LEAKY_WITNESS=(2.0,(1.1,0.0))` (~3.1e-3 miss) is a
  PROPERLY-ESCALATED flagged issue pinned as an escalation guard, not a defect.
- SPEC.md NOT updated though the plan expected it → Librarian doc-sync scope,
  not a code defect.

### Open doc-staleness carried from prior reviews (all → Librarian, verify fresh)
- INS-1-001/002/003 lineage (exterior_polar tag/2D-carrier) — believed CLOSED
  per 2026-08-15 note; re-grep before re-opening.
- Region vocabulary (lobe_exterior/lobe_interior/wedge_interior) absent from
  SPEC.md/DATA_CONTRACTS.yaml.
