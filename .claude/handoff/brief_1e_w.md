# Build Brief: 1e-w — Frequency Axis Collocation

## Mission

Per the collocation fragment: "The w axis is uniform in log w, which this
fragment calls a guess; the envelope varies on w * Delta_tau. This is the
one sub-build with a falsifiable oracle ALREADY in the tree: the serving
path's leave-one-out envelope refinement (_LOO_SEED_NODES = 8, stop 4e-3,
ceiling _LOO_MAX_NODES = 48) places nodes by measured error. The analytic
rule must REPRODUCE those node counts to within the LOO stopping tolerance;
if it does not, the analytic scale is wrong, because the LOO result is the
measurement."

The deliverable: either validate that uniform log-w is correct (and document
why), or replace it with a scale derived from w * Delta_tau.

## In scope

- Analyze whether the current uniform log-w grid is the correct scale for
  the envelope variation.
- The falsifiable test: compare the analytic grid's node count against the
  LOO refinement's measured node count. They must agree within the LOO
  stopping tolerance.
- If uniform log-w is wrong: replace with the correct w-scale.
- If uniform log-w is right: document WHY (what makes the envelope smooth
  in log-w rather than in w*Delta_tau).
- DRY test connecting the chart's w-axis to the envelope's actual variation
  scale.

## Out of scope

- Training (step 9).
- Other axes (already done).

## Constraints

- Fast tests only.
- Follow AGENTS.md and the spec/TODO workflow.
