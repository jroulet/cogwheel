# Build: trim saddle tube training to the D2 fundamental arc set

## Mission

Tube SERVING folds through the D2 gauge-image search
(`surrogate._tube_theta_inframe`: images theta, pi-theta, -theta,
pi+theta, identity first) — a trained arc serves its three mirrors. The
astroid side already trains only its fundamental arc (4 -> 1). The saddle
side still trains `arcs[:max_tube_arcs]` (production 20 = ALL 6 deltoid
arcs), so up to half the saddle tube charts duplicate what mirror images
already serve — symmetry-redundant cost the owner's directive forbids.
MEASURED at production config (tiling census, 2026-08-14, HEAD ff52e85):
tube:-1 trains 6 arcs / 61,740 nodes; the fundamental set should cut this
~2x. Extend the astroid's deterministic fundamental-arc selection to
parity -1.

## Facts (verify at HEAD)

1. Derive the saddle fundamental arc set FROM THE FOLD LAW, not by
   guessing a count: an arc is redundant iff its [theta_lo, theta_hi]
   interval is the image of a retained arc's interval under one of the
   three reflections (t -> pi-t, -t, pi+t). Compute the orbit partition of
   the 6 detected arcs; retain one representative per orbit (expect
   6 -> 3, but DERIVE it — the deltoid's cusp at the wrap and the
   orientation-reversing gauge<->source map (measured, see
   `_tube_training_arcs`'s comment) make interval bookkeeping subtle).
2. VERIFY SERVE COVERAGE with the image search ACTIVE, not by arc
   bookkeeping alone: after the trim, sweep the full theta ring (both
   parities, a dense grid of gauge angles at an in-band eta) and assert
   every angle either serves through some trained arc's frame via
   `_tube_theta_inframe` or falls in an inter-arc cusp gap — no NEW
   unserved interval vs the all-6 incumbent (measured comparison, not
   assumed).
3. The F079 topology guard (`_EXPECTED_ARCS = {1: 4, -1: 6}`,
   detected-arc count) is about the TILER and stays byte-unchanged; only
   trained-arc SELECTION moves. The census's Q1 then reports saddle
   detected 6 / trained <n_fundamental>.
4. `max_tube_arcs` currently governs the saddle slice; decide its fate
   explicitly: either it caps the fundamental set (min) or it retires for
   parity -1 with the TrainingConfig field comment updated — never a
   dead-but-misleading knob (the INS-2 lesson from the fold build).

## Scope

IN: `_tube_training_arcs` saddle branch (orbit-partition selection);
serve-coverage verification test (fact 2, fast synthetic config); census
Q1 expectation update if its sanity numbers move; TrainingConfig comment.
OUT: any training run; the tiler; the fold helper; lobe/deltoid far-field
regions (separate redesign fragment); the campaign.

## Acceptance

- Saddle trained-arc set = one representative per D2 orbit (derived,
  reported); tube:-1 census nodes drop accordingly (~2x, reported from a
  census re-run at production config — engine-free, driver can re-run).
- The theta-ring serve-coverage sweep (fact 2) shows NO new unserved
  interval vs the all-6 incumbent, both parities.
- Detected-arc guard unchanged (6/6 detected); full fast suite green.

## Constraints

Branch claude-dev; fragments (closes
`todo.d/lensing_saddle_tube_fundamental_training.md`, `[→ spec]`);
values-not-paths; in-build tests fast; no engine calls (this is geometry
+ chart bookkeeping); escalate rather than iterate on any surprise.
