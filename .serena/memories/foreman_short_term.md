# Foreman Short-Term Notes

- INS-5-DOC-1, SEVENTH consecutive pass routed to Foreman-Lite: re-verified
  via search_for_pattern that SPEC.md still has zero occurrences of
  "LensedMarginalizedExtrinsic". Finding text explicitly says
  "Librarian-owned... NOT a code defect" and "→ Librarian: add a SPEC
  row/sentence...". My hard ownership rule forbids Foreman-Lite from
  writing SPEC.md. Declined again; made NO changes (7x-confirmed no-op
  pattern, up from 6x last pass). This is now a persistent, unresolved
  orchestrator routing bug — INS-5-DOC-1 must be routed directly to
  Librarian, never to Foreman-Lite. Strongly recommend orchestrator add
  a routing rule keyed on "Librarian-owned" / "→ Librarian:" prefix in
  finding text to skip Foreman-Lite entirely for this finding class.
  Repeating this note every pass wastes a full agent turn each time;
  if this keeps recurring after this escalation, the routing bug itself
  should be treated as the higher-priority defect.