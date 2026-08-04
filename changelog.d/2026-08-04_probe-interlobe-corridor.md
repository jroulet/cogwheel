## 2026-08-04

### Measurement: interlobe corridor is geometrically negligible (region 2 closed)

`scripts/probe_interlobe_corridor.py` (402 lines) measures the inter-lobe
corridor geometry for the macro-saddle surrogate at
`gamma = 1.1, 1.3, 1.5, 2.0`:

- Corridor width / centroid-separation ratio: **6–17 %** depending on gamma.
- Area fraction of lobe interior captured by the corridor: **0.00 %** at
  all measured gammas.

**Conclusion:** the corridor exists geometrically but captures zero prior
mass from lobe interiors.  Region 2 ("interlobe corridor") in the
coverage-map TODO is **closed**; the exact-engine fallback handles the
negligible corridor with no accuracy or efficiency concern, and no
follow-up wiring is needed.

Commit: `de1aebd`
