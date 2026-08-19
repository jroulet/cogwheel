# Rung P operator series: reach vs truncation order (measured at 1a06ef3)

Engine-free.  Honest ceiling = largest `w` on a 40-point log grid up to the
draw's band top whose N/2N tail ratio over the SHIPPED `_operator_terms`
clears `CERTIFICATION_BAR`.  Sample: 90 astroid-side `engine_residual` draws
per region from `.claude/handoff/demand_census_post_born_10k.json` (10k draws).

| region | order | whole band | any coverage | median ceiling |
|---|---|---|---|---|
| exterior (pool 801) | 8 | 34.4% | 80.0% | 8.79 |
| | 12 | 41.1% | 95.6% | 4.06 |
| | 16 | 47.8% | 100.0% | 4.06 |
| | 24 | **61.1%** | **100.0%** | 5.29 |
| tube (pool 752) | 8 | 6.7% | 42.2% | 9.11 |
| | 12 | 8.9% | 52.2% | 8.25 |
| | 16 | 6.7% | 71.1% | 1.98 |
| | 24 | **22.2%** | **88.9%** | 1.81 |
| wedge_interior (pool 2682) | 8 | 1.1% | 10.0% | 6.61 |
| | 12 | 1.1% | 18.9% | 7.07 |
| | 16 | 1.1% | 31.1% | 3.64 |
| | 24 | **4.4%** | **48.9%** | 3.70 |

## Readings

- **The series is CONVERGENT, not asymptotic.**  Coverage rises monotonically
  with order in every region.  Raising the truncation order buys reach; it is
  not a fixed physical wall.
- **The exterior closes completely at order 16-24**: 100% of residual draws get
  at least partial analytic coverage, 61.1% get their WHOLE band.  The exterior
  is 54% of the planned campaign nodes.
- **The interior is NOT out of reach after all.**  At the shipped order 8 only
  10% of `wedge_interior` residual draws get any coverage, which reads as "Rung
  P does not work inside the caustic".  At order 24 it is 48.9%.  The earlier
  write-off was an artifact of the truncation order, not the physics.
- **Median ceiling is a MIX statistic, not a per-draw one.**  It falls at higher
  order because newly certified draws (which have low ceilings) enter the
  median.  Do not read a falling median as lost reach; read `any coverage` and
  `whole band`, which are monotone.
- Cost: order `M` evaluates `2M` operator terms of pure special-function work
  per certificate call — no engine evaluation.  Weigh the order against
  certificate call cost, not against engine cost.
