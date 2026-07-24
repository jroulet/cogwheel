# Build 8h-b3-fin — Complete the caustic-fixed core (S1-1 partial in tree)

## Mission

Complete Build 8h-b3, whose S1-1 coder exhausted an honest 90-turn
budget mid-migration (infrastructure now hardened: coders get one
bounded continuation on exhaustion — this build benefits from it).
S1-1's PARTIAL WORK is UNCOMMITTED in the tree: r_caustic helper in
chang_refsdal/geometry.py, chart-schema changes in surrogate.py, and a
mid-flight tiler migration in surrogate_training.py. The approved plan
is BINDING and preserved: read
`/tmp/build8hb3_caustic_core_approval/plan.json` in full — its six WPs
(S1-1 exterior coordinate migration, S1-2 w-windowed labels + tags,
S1-3 fixed w-windows + component grids, S2-1 interior directional
admission, S2-2 per-lobe saddle interiors, S2-3 whole-interior SACR-C
label), Professor Inputs, and Domain Test Descriptions are the spec.
Do NOT re-plan; verify the in-tree partial S1-1 against its WP text,
complete it, then proceed through the remaining WPs in dependency
order. The tag contract remains the named top risk (a tag mismatch
silently reconstructs the WRONG F).

## State of the tree (measured)

- Uncommitted partial S1-1 across: cogwheel/lensing/chang_refsdal/
  geometry.py (r_caustic), cogwheel/lensing/surrogate.py (axis
  schema), cogwheel/lensing/surrogate_training.py (tiler,
  INCOMPLETE — the exhausted coder's last edit was here). `git diff`
  is the source of truth; the coder change report was lost.
- Everything committed through b14df4b (ghost kernel) is stable
  underneath: map v2 with ceilings, band-split dispatch, ghost
  primitive with Im tau_c exposed, 8g/8g-b/8h-a test batteries.
- All measured facts and width rulings of the original brief
  (.claude/handoff/lensing/build8hb3_brief.md) stand; the Born rung
  is deferred to its own build (owner-sequenced BEFORE the campaign);
  the prior-dependence caveat recording goes to the post-gate
  Librarian.

## Out of scope — hard fences

Identical to the original brief: NO quad-double, NO tube changes, NO
campaign in-build, NO tolerance changes, NO Pearcey model, ghost only
per its measured gate. NO SDK/infra changes (the continuation fix is
already in the tree; commit it with the build).

## Acceptance (two-tier)

1. In-build (FAST): the approved plan's per-WP verification clauses
   and Domain Test Descriptions verbatim, including the seam tests
   with their reachable-reds, the whole-interior three-gamma
   falsification grid, strata-removal byte-identity, the tag-contract
   mixed-artifact test, and tube byte-identity; fast tier green.
2. POST-BUILD (driver): calibration re-pilot (cost-quoted) vs the P2
   before-numbers; serving census (the climb from 2.2% must begin);
   then Born rung, qd-or-GLoW (feasibility probe running), map
   extension, the ONE campaign, ladder census.
