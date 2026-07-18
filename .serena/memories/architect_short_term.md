# Architect Short-Term Observations

Build 3g (lensing ratio layer) planned 2026-07-18:
- Goal: warm lnlike 29.5ms -> <=10ms by candidate/fiducial RB on the lens
  sector. Engine cost is entirely E(w) LOO nodes (~30-44 @0.41ms). Ratio
  rho_bare(w)=exp(iw*dtau_c)*E_cand/E_fid needs ~8-12 nodes.
- Professor code-pins: share (m_lens,z_lens) exactly (so w=w0); snap ALL
  five geometry params to lattice (gamma .03, beta pi/16, kappa .02, y .05);
  fiducial = PURE function of candidate -> memoize per lattice key
  (determinism preserved). dtau_c = tau_c_cand - tau_c_fid pulled out
  analytic; reconstruct E_cand=exp(-iw*dtau_c)*rho_spline*E_fid_spline ->
  existing reconstruct_from_envelope with CANDIDATE geometry/critical_delay.
- Simplifier TRIM: drop topology-aware caustic-cell partitioning; use two
  one-line guards -> fallback to (certified) direct SACR-C: image-count
  mismatch (real_mask.sum differs) OR health min|E_fid|/max|E_fid|<0.01.
  Fiducial-side LensDomainError/CancellationError -> fallback; candidate-side
  refusals propagate (symmetry w/ bruteforce). No channels.py change; no new
  public API; no new constructor param (spacings are module constants).
- Tolerances (Professor authority): identity at lattice point env<1e-13,
  lnlike<1e-9 nats; ratio-vs-direct <0.1 nats; ratio-vs-bruteforce inherited
  max(1.5,1e-2|bf|); LOO stop 4e-3 unchanged. 10ms is HARD gate, step-rule
  only via Professor obstruction; surrogate 2D E_fid table is named backstop.
- Two coder WPs: WP1 behavior-preserving refactor (extract direct path +
  _kernels_from_dense_envelope seam + snap/key helpers + _fid_cache);
  WP2 ratio layer + guards + fallback + testing seam _force_direct.
