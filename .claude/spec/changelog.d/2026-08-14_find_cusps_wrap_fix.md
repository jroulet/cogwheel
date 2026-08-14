---
date: 2026-08-14
---

Fixed F079: `_find_cusps` computed cusp dip-window spans with linear angle
arithmetic across the periodic wrap, silently dropping both arcs adjacent
to theta = 0 (half the astroid fold ring had no tube chart). Spans are now
wrap-aware; `detect_caustic_structure` cross-checks surviving ARC count
(4 astroid / 6 saddle), not just cusps. The inert, wrong-units cusp-arm
coverage constants and their `_tube_serves` window shrink are retired; the
tube gate excludes on the full cusp window.
