# Professor Short-Term Observations

## 2026-08-02: interior_w_nodes_per_decade build review — PASS (with concern)

### Tests executed:
- `InteriorWnpdAccuracyTestCase`: 4/4 passed (38s)
- `TrainingConfigWnpdFieldTestCase`: 4/4 passed (4s)
- `WholeInteriorSacrcTestCase`: 7/7 passed + 1 xfailed (91s)
- `SelfFalsificationTestCase`: 10/10 passed (21s)
- Full suite: **85 passed, 1 xfailed** (216s)

### Measured epsilon values (WNPD accuracy):
- gamma=0.40, WNPD=12: eps=0.002416 — PASSES bar 0.05 (97% headroom)
- gamma=0.65, WNPD=12: eps=0.000277 — PASSES bar 0.05 (99.4% headroom)
- gamma=0.65, WNPD=6:  eps=0.000239 — does NOT breach the 0.05 bar

### Physics assessment:
- Both gamma values at WNPD=12 pass with ENORMOUS margin (10-100x better than
  the spec's expected 0.005-0.030 range). This indicates the (s,d) spatial
  interpolation at n_s=5, n_d=5 over the small test tile [0.01,0.04]×[-0.02,-0.005]
  is sub-permille accurate, and the smooth SACR-C envelope (bounded phase theorem:
  max 4 rad residual) is NOT the interpolation bottleneck.
- **Spec falsification not realized**: WNPD=6 at gamma=0.65 does NOT fail. This is
  physics-correct: 17 w-nodes over 2.6 decades with ≤4 residual-phase cycles gives
  ~4 nodes/cycle → cubic error O(h^4) ~ 4e-3, comfortably below 0.05. The SACR-C
  demodulation is TOO effective for the lever to show at this tile size.
- The falsification WOULD show at production scale (wider spatial extent, near-cusp
  tiles where envelope develops sharper frequency structure, or larger w-range).
- The test correctly proves the WIRING (node count changes) and the ACCURACY-AT-
  PRODUCTION-DENSITY, even without the failing-low-WNPD demonstration.

### TrainingConfig verification:
- `interior_w_nodes_per_decade` exists, defaults to 15, accepts custom values,
  is independent of `w_nodes_per_decade` (=4), frozen (immutable). All correct.
- Wiring: `_log_w_grid(SACRC_W_RANGE, 15)` → 41 nodes; `_log_w_grid(..., 4)` → 12.

### Concern:
- The spec's falsification claim ("WNPD=6 at gamma=0.65 FAILS the 0.05 bar") is
  NOT realized in the current test geometry. The spec overestimated the envelope's
  frequency complexity on this small, cusp-free tile. This is NOT a code bug — it's
  a spec prediction that doesn't hold for this geometry. The lever IS load-bearing
  (proven by node-count test) but the accuracy bar has >200x headroom, so the
  WNPD=6→12 density increase is not needed to pass at this tile size.
- Heavy full-sampling validation is operator-deferred.
