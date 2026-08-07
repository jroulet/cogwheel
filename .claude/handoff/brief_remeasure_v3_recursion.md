# Build Brief: Re-measure under current code (wedge v3 + exterior recursion)

## Mission

Two measurement/verification items, both unblocked by the 2026-08-07
`subdivision_recursion` build (bca9534). Both re-run probes that were
measured against now-stale code:

1. **Wedge probe charts under schema v3** — the 18-chart / median 5.47e-4
   interior result and the 13/16-children subdivision result were measured
   against v2 charts, which hard-refuse under the v3 schema. The conclusions
   are "very likely unchanged" but unmeasured.

2. **Exterior recursion effectiveness** — bounded recursion shipped for the
   exterior but was only measured on the interior. 35 of 57 exterior charts
   failed the 1e-3 bar; the hypothesis is that every marginal tile got one
   halving and was abandoned. Need to measure: how many of the 35 now clear
   1e-3, the achieved-depth histogram, and whether any tile hits depth-3.

## Measured facts (from the fragments)

- Wedge schema v3 requires `theta_to_u`; v1/v2 artifacts hard-refuse.
  The stored `u = d**(2/3)` array is identical between v2 and v3 — only the
  field name changed. Serve is coordinate-agnostic through the stored map.
- Exterior training: 84% of exterior charts are subdivision children, 35/57
  failed the 1e-3 bar. Interior measured: 13/16 children cleared at one
  halving, three marginal (6.50e-2, 6.70e-2, 5.95e-2 against 5e-2 bar).
- The 2026-08-06 probe scripts live under the scratchpad (git history);
  recover or reproduce them.

## What to do

### Part 1: Wedge v3 retraining
- Re-run the interior wedge probe against v3. Confirm the 18-chart /
  median 5.47e-4 / ~10.5 min numbers are reproduced (or note the delta).
- Record the achieved subdivision depth histogram.

### Part 2: Exterior recursion
- Rerun exterior training for ONE band with recursion live.
- Report: (i) how many of the 35 previously-failing charts clear 1e-3,
  (ii) achieved-depth histogram, (iii) whether any tile hits depth-3.
- A depth-3 cap hit is evidence the COORDINATE is wrong, not the cap —
  route such tiles to the polar re-chart fragment, don't deepen the cap.

## Acceptance

- Both probes re-run under current code; numbers recorded in a findings
  fragment or the completed.d entry.
- If the wedge v3 numbers reproduce: the interior eps acceptance and the
  recursion cap justification stand on fresh measurement.
- If the exterior recursion clears a meaningful fraction of the 35: the
  polar-vs-(s,d) A/B can compare like with like.

## Constraints

- Fast/measurement work — no new coordinate design. This is the
  prerequisite for the polar re-chart (the hub item).
- Follow AGENTS.md and the spec/TODO workflow. Close the two fragments
  with a completed.d entry and re-render.
