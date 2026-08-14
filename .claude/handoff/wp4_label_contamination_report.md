# WP-4 report: retroactive label-contamination check (REPORT ONLY)

Scope: determine whether the two shipped training artifacts drew any labels
through the exterior positive-parity fold arm
(`operator._positive_parity_grid` -> `channels.evaluate().exact_total` / `F_op`)
in the contaminated band `60 < w <= 150` at exterior (positive-parity,
`|y|/r_caustic > 1`) cells — the band where the fold arm shipped the wrong
value before WP-1 (see `.claude/handoff/fold_exterior_ghost.md`, F075).
No retraining, no artifact edits were performed.

Method: read each producer's oracle route (how the tabulated label value is
obtained) and cross-reference the shipped `.npz` provenance for the actual
grid axes. Node values below were computed with the shipped-map constants
(`ASTROID_WALL=443.7`, `SADDLE_WALL=58.0`, `_w_nodes` at 12 nodes/decade on
`[1, wall]`).

---

## Summary verdict

| Artifact | Oracle route | Verdict |
|---|---|---|
| `certified_ppgo_map.npz` (2026-08-03) | `ChangRefsdalChannels(w_prefix).evaluate(...).exact_total` as the reference `F` in `ppgo_map._measure_cell` | **CONTAMINATED** at 32 positive-parity exterior cells |
| `born_residual_chart.npz` (2026-08-04) | `ChangRefsdalChannels(w_grid).evaluate(...).exact_total` in `scripts/train_born_residual.py` | **CLEAN** (w-grid tops out at 60.0; never enters `60 < w`) |

---

## 1. `certified_ppgo_map.npz` — CONTAMINATED

### Oracle route (the contaminated path)

`build_map` -> `_measure_cell` places a source at
`rho_center * reach * (rot @ direction)` for each cell and, inside the
per-angle `evaluate(k)` closure, computes the reference `F` as:

```python
partition = ChangRefsdalChannels(w_prefix).evaluate(
    gamma=gamma, y=source, beta=0.0, kappa=kappa)
...
error = np.abs(exact - ppgo) / denominator      # exact = partition.exact_total
```

`partition.exact_total` **is** the F075 label oracle. Per handoff fact 2,
in `60 < w <= 150` at exterior 2-image (positive-parity) configs
`channels.evaluate().exact_total` returns the fold-arm value EXACTLY (the arm
is offered by `_positive_parity_grid` before `f_schwinger` with no accuracy
gate), which errs 15–216 % (fact 1). So the certification error
`|exact - ppgo| / max|exact|` for those cells at those w-nodes was measured
against a WRONG reference.

### Which w-nodes are contaminated

Positive-parity sweep uses `_w_nodes(443.7)` = `np.geomspace(1.0, 443.7, 33)`.
Five nodes land in the open band `60 < w <= 150`:

```
w in {66.05, 79.91, 96.68, 116.96, 141.50}
```

These are far below the astroid wall (443.7), so they sit inside the accepted
w-prefix of every exterior positive-parity cell (`_max_accepted_prefix` does
not truncate them — the fold arm ships a value rather than refusing), and they
therefore feed both `error` and the `denominator = max|exact|`, and hence the
sup-over-w floor `w_cert`.

### Which cells are suspect (parity=positive, w x |y|/rc x gamma)

Exterior rho-bands (`_rho_center >= 1.0`), from
`rho_edges = (0.0, 0.5, 0.9, 1.0, 1.5, 2.5, 4.0, inf)`:

| ri | rho band | rho_center (sample) | rho_measured_max |
|----|----------|---------------------|------------------|
| 3  | [1.0, 1.5] | 1.25 | 1.5 |
| 4  | [1.5, 2.5] | 2.00 | 2.5 |
| 5  | [2.5, 4.0] | 3.25 | 4.0 |
| 6  | [4.0, inf] | 6.00 | 6.0 |

Positive-parity gamma-bands (`hi <= 1.0`), from `_gamma_band_edges()`,
`gamma_center = sqrt(lo*hi)`:

| gi | gamma band | gamma_center |
|----|-----------------------|-----------|
| 0  | [0.05,     0.091028] | 0.0675 |
| 1  | [0.091028, 0.165723] | 0.1228 |
| 2  | [0.165723, 0.301709] | 0.2236 |
| 3  | [0.301709, 0.45]     | 0.3685 |
| 4  | [0.45,     0.549280] | 0.4972 |
| 5  | [0.549280, 0.55]     | 0.5496 |
| 6  | [0.55,     0.9]      | 0.7036 |
| 7  | [0.9,      1.0]      | 0.9487 |

**Suspect cell set: the full cross product** =
positive parity x {8 gamma bands gi=0..7} x {4 exterior rho bands ri=3,4,5,6}
= **32 cells**, each measured with the contaminated reference at
`w in {66.05, 79.91, 96.68, 116.96, 141.50}`.

Suspect cell coordinates `(parity, gamma_center, rho_center)`:

```
positive, gamma in {0.0675, 0.1228, 0.2236, 0.3685, 0.4972, 0.5496, 0.7036, 0.9487}
        x rho_center in {1.25, 2.00, 3.25, 6.00}
```

### Cells that are NOT contaminated by this bug (clean)

- **All saddle-parity cells** (parity code 1). Saddle sweep uses
  `_w_nodes(58.0)`, whose top node is 58.0 < 60 — the sweep never enters
  `60 < w`. Additionally the bug is a `_positive_parity_grid` phenomenon and
  the saddle grid is a different macro-saddle route. Clean by w-range.
- **All positive-parity INTERIOR cells** (rho-bands ri=0,1,2, centers 0.25,
  0.70, 0.95 < 1.0). The fold-arm defect is an EXTERIOR 2-image phenomenon;
  interior configs are 4-image and route through the correct path (the
  interior 4-image fold path is explicitly untouched by WP-1). These cells
  sample the (60,150] w-nodes but with an uncontaminated `exact_total`.
  (Their separate interior ppGO certificate is out of WP-4 scope.)

### Likely direction of the error (advisory, not re-measured)

The floor rule is the smallest w above the last upward re-crossing of the
`CERTIFICATION_BAR = 1e-4`. The contaminated error (~0.45, fact 1) is far
above the bar across the whole (60,150] band, so the last re-crossing — and
thus `w_cert` — is pushed up to ~141–150 (or the cell is driven toward
`STATUS_BEYOND_WALL`). This is the OVER-conservative direction: the shipped
map will refuse to serve ppGO below a w where ppGO is in fact fine, forcing
consumers to fall back to the exact engine over more of the exterior band.
It is a coverage / performance loss for the 10 downstream consumers, **not** a
correctness risk (the map never over-certifies here). The exact per-cell floor
shift requires re-measuring with the WP-1-corrected `exact_total`; that is a
retraining action, deliberately NOT performed here.

---

## 2. `born_residual_chart.npz` — CLEAN

### Oracle route

`scripts/train_born_residual.py` labels each grid point with
`F_exact = partition.exact_total` from `ChangRefsdalChannels(w_grid).evaluate`
(same oracle family as the ppGO map) minus the demodulated Born carrier.

### Why it is clean: the w-grid never enters the contaminated band

```python
w_grid = np.geomspace(5.0, 60.0, 10)   # max node = 60.0
```

The top node is exactly **60.0**. The contaminated band is the OPEN interval
`60 < w <= 150`, so no grid point lies in it. Handoff fact 2 states `w <= 60`
is the safe exact-DD-batch regime, below the band where `_positive_parity_grid`
offers the fold arm. Even though the chart's cells are far-exterior,
positive-parity (`rho in [2.0, 4.0]`, `gamma in [0.05, 0.9]`, `theta = pi/4`) —
i.e. exactly the geometry that WOULD trigger the exterior fold arm at higher w —
the arm is never reached at `w <= 60`, so every `exact_total` label is the true
DD value. No suspect cells.

---

## Consumer / ordering note

Per the pipeline graph, `certified_ppgo_map` has 10 consumers and
`born_residual_chart` has 1. The 32 flagged ppGO cells are a **downstream
retraining advisory** only; no consumer is edited and no artifact is
retrained in this build. Recommended follow-up (separate build): re-run
`scripts/train_ppgo_map.py --production` after WP-1 lands so the exterior
positive-parity floors are re-measured against the corrected (ghost-served /
refuse-to-engine) `exact_total`, then re-hash. `born_residual_chart.npz`
needs no action.
