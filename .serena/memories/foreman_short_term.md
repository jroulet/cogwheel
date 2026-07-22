INS-1-003: finding text explicitly says "→ Librarian: update the SURROGATE
training narrative in SPEC.md ... bump spec_version. Triage as spec-sync,
not a code bug." This is a Librarian-owned SPEC.md edit, not a Foreman-Lite
fix. Declined per hard ownership rule (Foreman-Lite must not write
SPEC.md/TODO.md/COMPLETED.md/CHANGELOG.md). No files touched this session.
This is the same recurring mis-route pattern noted in foreman_knowledge
(INS-5-DOC-1 recurred 7x) — worth flagging to orchestrator that
Librarian-tagged findings keep landing in the Foreman-Lite queue instead
of being routed directly to Librarian.