---
section: Backlog
---
# Envelope surrogate + micro-levers — close the lensed/unlensed per-eval gap [→ spec]

Owner directive (2026-07-18): the lensed likelihood at ~9.8 ms/eval is
still ~10-100x the unlensed RB cost, and the lensed posterior explores
MORE dimensions (4 extra sampled lens params + lens-sector structure) —
a double whammy on total sampling cost even after extrinsic
marginalization. The surrogate is therefore a standing objective, not a
one-time backstop; do not let it fall out of planning.

Current cost structure (post-Build-3g, measured): per-proposal engine =
~8 ratio nodes x ~0.4 ms + amortized fiducial builds (~30-44 nodes per
new lattice cell); plus caustic search ~1.9 ms and data/norm contraction
~2.5 ms. Levers, in rough order of value:

1. **E_fid surrogate (per-lattice-cell)**: tabulate the fiducial
   envelopes — ONE smooth beat-free 1D curve per lattice cell — killing
   both the fiducial-build cost and most of the ratio-node engine cost.
   Any GLOBAL table must respect the w = xi(M_L, z_L)*f moving-grid
   constraint; the per-cell form sidesteps it.
2. **Micro-levers** (fenced out of Builds 3d-3g, still on the shelf):
   nearest-caustic Newton shortcut (~1.9 -> ~0.3 ms, geometry.py,
   value-preserving + branch-invariant obligations) and weight-vector
   contraction fusion (~2 -> ~1 ms, operator.py, refusal quantities
   byte-unchanged, F005-style re-certification).
3. **Schwinger-cost coupling**: the negative-parity builds replace the
   shear series with the exact both-parity Schwinger representation
   (see negative_parity_research.md). Its per-point cost is UNMEASURED;
   if it lands slower than the current ladder path, the surrogate's
   value rises accordingly (surrogate of the Schwinger-based envelope
   over the enlarged domain). Measure per-point cost in those builds
   and revisit this fragment's priority.

Target: lensed per-eval within a small factor (~2-4x) of unlensed RB.
Sequencing per owner: after Build 5 (extrinsic marginalization) and
alongside/after the negative-parity builds. Oracles and gates follow
the house rules (engine as F002-clean oracle; zero false accepts; no
tolerance widening; fallback-to-exact preserves certified-or-refuse).
