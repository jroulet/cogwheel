# Professor short-term checkpoint (2026-07-18, negative-parity commission)

Commission COMPLETE.  Deliverable written:
`.claude/handoff/lensing/negative_parity_research.md` (verdict: the
treatment exists; one precisely characterized obstruction inside it).
Durable physics folded into `mem:professor/microlensing_chang_refsdal`
(new saddle section).  Engine untouched (ec8a276); all numerics in
session scratchpad `np_exp1..9*.py`, interpreter cogwheel-newlal.

Headlines (all measured, details + tables in the handoff file):
- Geometry layer parity-agnostic: quartic finds saddle-domain images
  unmodified; census 2:(1,1), 4:(0,1,1,1) (index theorem, 4000-source
  scan, 0 anomalies); critical curves = same v(theta) formula with +-
  branch -> two lobes; caustics = two 3-cusp deltoids.
- Operator shear series DIVERGES for gamma'>1 (radius = parity
  boundary; best truncation error O(1) at all w) - hard obstruction.
- Replacement: exact 1D Schwinger representation (derived; both
  parities); validated vs independent 2D lens-plane mpmath oracle to
  2.2e-15 on the saddle domain, vs F_op 4e-15 positive parity, vs
  point-mass closed form exactly.  Single y-INDEPENDENT cancellation
  channel L_S = pi*w/4 (measured e^{pi w/4}*1e-16 law); float64 ceiling
  w~18-30, dd ceiling w~64 (matches engine's 60-band).
- Deep band: F -> e^{-i pi/2}/sqrt(gamma^2-lam^2), |F| correction O(w),
  phase drift w[tau_G + (1/2)ln(w/2) + c0] modeled to 1e-3.
- SACR-C carries over (two-lobe nearest-caustic carrier): greedy
  N = 20-25 over 15 saddle configs incl. fold/cusp crossings
  eta=+-0.002 (positive-parity band 19-26).  max|S H| <= 1.46 on
  crossings, <= 2.8 on random scan.
- Scratch traps found: t_min/carrier demodulation mismatch fakes the
  beat disease (N 72 vs 24); float64 Schwinger truth silently garbage
  at w~69 (e^{pi w/4} law) - both recorded as build lessons.
- Build shape proposed: two sequential builds (S1 engine geometry +
  dd-quadrature wave branch; S2 channels/likelihood/prior), gates
  listed in the handoff Sec. 11.  FINDINGS addenda list in Sec. 10.

Open/unclaimed: v-plane steepest-descent evaluator (saddles of the 1D
integrand = image quartic roots v*=u) documented but not built;
lam <= 0 (Type III) named refusal only; c0 constant in deep phase not
derived in closed form.
