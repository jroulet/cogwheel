# Build Brief: Inter-lobe corridor probe (coverage region 2)

## Mission

The saddle parity has two deltoid lobes. `LobeInteriorChart` serves draws
inside each lobe, and `FarFieldChart` serves the exterior. But draws in
the CORRIDOR between the two lobe centroids — inside the caustic but not
inside either lobe — may fall through.

## Task

Write `scripts/probe_interlobe_corridor.py` that:

1. For representative saddle gammas (1.1, 1.3, 1.5, 2.0):
   - Compute the two deltoid lobe centroids
   - Sample 100 source positions along the line between them
   - For each, call `select_chart` and record what serves (or doesn't)

2. Report:
   - What fraction falls through to exact engine?
   - If any are served, by which chart type?
   - If none are served, what's the geometric extent of the gap?

3. If a gap exists: is it physically small (corridor width << caustic size)?
   Does it matter for the prior (what fraction of prior draws land there)?

## Acceptance

- Measurement runs and produces a served/unserved map of the corridor.
- If gap is negligible (<1e-3 of prior mass): mark region 2 inter-lobe
  corridor as CLOSED with the measured number.
- If significant: document the geometry and file a follow-up.

## Constraints

- Pure measurement, no code changes.
- Fast (geometry + select_chart calls, no full training).
- Follow AGENTS.md and the spec/TODO workflow.
