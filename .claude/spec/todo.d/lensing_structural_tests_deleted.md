---
section: Backlog
---
- **Restore the surrogate structural tests once the serving schema settles**
  `[housekeeping]` — three classes were DELETED from
  `cogwheel/tests/test_lensing_surrogate_training.py` because they pinned the
  training-record structure while that structure is mid-redesign:

  | class | covered |
  |---|---|
  | `TilingRecordTestCase` | tiles pairwise disjoint; tiles wholly outside the exclusion rho; every stratum recorded even at zero tiles; region-cap truncation records its dropped count; tiling diagnostic plot |
  | `WholeBandContainmentTestCase` | a chart's w-range contains every in-stratum draw's whole detector band; cap-truncated corners are recorded, not silently served |
  | `ResidueBucketPartitionTestCase` | prior draws partition into exactly chart-served / beyond-w-cap / residue, with no double-count and no silent drop |

  **Restore with** `git show edf8d485a564f744f61a5b9455739519d9160f31 --
  cogwheel/tests/test_lensing_surrogate_training.py` (the commit immediately
  before the deletion; the classes are present and green there, already ported
  to caustic-fixed coordinates and the anisotropic tile boxes).

  WHY DELETED rather than kept-and-gated: they are bookkeeping over a schema
  that changed three times in recent history (8h-b3 caustic-fixed axes, 8h-b4
  per-column admission, S1-3 region windows replacing per-stratum exterior
  partitioning), and each migration silently killed them. Left in the tree,
  even skipped, they read as a specification -- an agent editing tiling code
  contorts production to keep them green and anchors the design to the shape
  we intend to leave. Git is the archive.

  WHAT THEY STILL GUARD, and where it must be re-established: the
  `ResidueBucketPartition` invariant (no double-count, no silent drop) is the
  accounting the coverage census rests on, so the CENSUS RUN must assert it
  directly -- that is the honest home for it, where the number is real rather
  than smoke-scale. The disjointness and outside-the-exclusion invariants
  should come back as fast structural checks once the record schema stops
  moving.

  WHEN: after the ladder closes and the serving design stabilises. Do not
  restore them mid-redesign.
