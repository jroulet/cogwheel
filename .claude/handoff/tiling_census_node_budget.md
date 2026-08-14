# Build: engine-free tiling census + node-budget predictor (pre-campaign gate)

## Mission

The pre-explosion guard class for the training campaign, re-instantiated:
an ENGINE-FREE census of what the trainer would build — tiling and node
allocation are pure geometry — with two-sided expected bands per
(region x parity): n_arcs, n_tiles, n_nodes. Explosion above the band,
SILENT-EMPTY below (the low-mass exterior once registered ZERO charts
silently). Required GREEN before the training campaign launches; its
output (sum over tiles x nodes) is the campaign's call-count estimate.

## The four questions (from the program fragment, updated to current HEAD)

1. **Astroid tube census AFTER the F079 wrap fix**: 4 arcs (count ARCS,
   not cusps — `_EXPECTED_ARCS` now enforces 4/6 in the tiler; the census
   asserts the TRAINER's per-arc chart/tile/node counts against bands).
   Measured windows at HEAD: 0.094/0.141/0.141/0.236 rad at gamma
   0.2/0.5/0.7/0.9; theta=0 window bit-identical to its theta=pi partner.
   NOTE the D2-fold build (`lensing_tube_d2_fold`, may land before or
   after this one — check HEAD) changes the CHARTED arc count to
   fundamental-domain arcs only; the census must count against whichever
   serving design is live and say which.
2. **The DELTOID far-field tiling in the additive scalar-reach gauge** —
   the transverse cone's envelope varies with DIRECTIONAL eta, not the
   tile coordinate: the same trap as the old astroid exterior. Likely
   needs an eta-adapted per-lobe radial axis (the log(rho-1) fix's
   deltoid analogue). THIS is the question most likely to force a
   coordinate redesign — find out BEFORE spending engine time. If the
   census shows tile counts exploding or nodes mis-allocated against the
   directional-eta measure, the verdict is a redesign fragment, not a
   bigger budget.
3. **Near-cusp tiles under the corrected F074 control coordinates**: is
   the old cusp-ray spline kink representable now? Engine-free check:
   tile the near-cusp annuli in the Pearcey controls and verify the
   proposed tiles' coordinate ranges are kink-free (monotone, bounded
   Jacobian) — geometry only.
4. **Per-region w-bands against the serve floors**: certificate floor
   (the c3 gate admits per-draw at its bar — state the effective floor it
   implies per region from the calibration data), farfield_w_floor,
   SADDLE_WALL = 58, the F074 w-floor 49, tube DD caps (33.1 at gamma
   0.5 / 18.3 at 0.7). Every region's trained w-band must sit consistent
   with what actually serves below/above it; a band that trains nodes
   nothing will ever serve is budget burned.

## Deliverable

`scripts/tiling_census.py` (or extend an existing census script — check
first; DRY): per (region x parity) a table of n_arcs / n_tiles / n_nodes
vs LO and HI expected bands, exit nonzero outside any band; plus the
campaign call-count estimate (sum tiles x nodes x labels-per-node) in a
machine-readable JSON the 7a cost estimate consumes. Fast suite test that
runs the census at a small synthetic config and pins the invariants that
survive refactors (bands themselves are report evidence, not pins).
PROBES ARE THIN CALLERS of the production tiler/subdivider (standing
rule; a parallel implementation is the F-class defect this program keeps
finding).

## Scope

IN: the census script + JSON output + fast test; the four questions
answered with numbers; a redesign fragment IF question 2 forces one.
OUT: any engine call; any training; fixing what the census finds (each
finding becomes its own fragment/build); the campaign itself.

## Acceptance

- Census runs engine-free in ~minutes, green at HEAD, bands two-sided.
- The four questions each have a written answer with numbers in the
  report; question 2's verdict explicitly states redesign-needed or not.
- Call-count JSON exists and 7a's brief can cite it.

## Constraints

Branch claude-dev; fragments; thin-caller probes; values-not-paths;
no engine calls anywhere in the census path (assert it — a census that
quietly calls the engine is the defect class itself).
