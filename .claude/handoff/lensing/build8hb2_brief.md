# Build 8h-b2 — Ghost-kernel machinery (complex-saddle pair, analytic continuation)

## Mission

Build the physics primitive the caustic-fixed core build (8h-b3)
consumes: extraction of the decaying complex-image ("ghost") pair from
the lens equation's quartic, and its analytically-continued kernel and
delay. SINGLE work package — this is the most delicate numerics in the
program (complex-log and square-root branch choices) and lands alone,
verified against an independent oracle, before anything consumes it.
The approved plan text is BINDING: WP3 of
`.claude/handoff/lensing/build8hb_plan_full_v1.json` verbatim — read
it in full (What/Where/How and the Professor Inputs on the ghost
construction). Summary of the pinned construction (the plan file is
authoritative where this summary compresses):

- The ghost pair is the complex-conjugate quartic-root pair that
  `geometry._generic_candidates` discards at its imag-tolerance cut;
  the u_c -> x_c map is complex-analytic, so continuation is exact.
  Select the DECAYING member by Im tau_c > 0.
- A DEDICATED `_ghost_kernel(x_c, matrix)` — never reuse
  delay/image_kernel/morse_index (real-only ops): delay via the
  complex Fermat potential with BILINEAR products and a complex log
  branch pinned by continuity from the real fold; amplitude
  1/sqrt(det H_c) on an unwrapped-arg branch seeded at the real
  saddle (the -i pi/2 Morse phase is ABSORBED in that branch —
  calling morse_index double-counts); C1/C2 series continued
  verbatim. Expose Im tau_c for the future gate.
- Named exceptions + shape/branch-continuity guards at the
  degenerate near-fold limit.

## Measured facts (pre-answered — do not re-derive)

- P1 probe anchors (2026-07-23, scripts preserved in the session
  scratchpad; numbers binding): at (gamma=0.2, rho=1.1, diagonal),
  w=8.5: |E_ff|=0.110, |C|=0.111, arg(E/C)=1.5 deg; R/E = 0.038 for
  w>=8. At (gamma=0.4, rho=1.1, diagonal), w=3.3: |E|=0.051,
  |C|=0.054, arg=-0.7 deg; R/E=0.060 for w>=3. The DECAYING Stokes
  branch (arg(det H) in (0,2pi)) wins at every config. On-axis
  (cusp-adjacent): Im tau_c = 0 exactly — the ghost is pure
  oscillation there (no decay); the kernel must still EVALUATE
  finitely (the gate that refuses to USE it there is 8h-b3's job,
  not this build's).
- At rho=4 the ghost is exponentially negligible (|C|max 7.5e-4 vs
  |E_ff|max 2.1e-3, Im tau_c=10.5) — the far sanity anchor.
- The engine's own conventions to match: t_min demodulation as in
  ppgo_map._measure_cell; dimensionless w, tau.

## Out of scope — hard fences

- NO label change, NO serving change, NO trainer change, NO gating
  logic (all of that is 8h-b3 consuming this primitive). geometry.py
  gains the extractor + kernel; nothing else in cogwheel/ changes
  behavior.
- NO map/certification changes. NO campaigns or sweeps.
- The existing real-image paths (find_images, image_kernel, delay,
  morse_index) stay byte-identical.

## Acceptance (two-tier)

1. In-build (FAST):
   (a) independent-oracle branch test (the v1 plan's spec, binding):
   ghost tau_c and amplitude agree in MAGNITUDE AND PHASE with a
   re-derivation via direct complex quartic roots + finite-difference
   complex Hessian determinant, computed without the module-under-test
   ghost functions (AST guard) — this catches branch-cut and
   Morse-double-count bugs;
   (b) P1-anchor reproduction: at the two anchor configs the ghost
   kernel reproduces the measured |C| and phase against E_ff within a
   few percent (numbers above), and the decaying-member selection
   picks Im tau_c > 0 at every off-axis config;
   (c) the on-axis Im tau_c = 0 case evaluates finitely with the
   correct pure-oscillation character (no NaN, no spurious decay);
   (d) rho=4 sanity: ghost magnitude negligible per the anchor;
   (e) real-image path byte-identity; fast tier green (tree gate).
2. POST-BUILD (driver): the low-w floor probe (~20 exact evals,
   minutes — measures whether the deep-diffraction correction varies
   on caustic or Einstein scale; decides the Born-rung question and
   feeds the 8h-b3 plan); then the 8h-b3 core brief.
