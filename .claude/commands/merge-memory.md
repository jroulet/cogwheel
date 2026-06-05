Semantic 3-way merge for a Serena memory file. This is called by
`scripts/mind_meld.sh` (in `triage` or `resolve` mode) during a `MIND_MELD=1`
merge to avoid the concatenation + duplicate-section artifacts that git's
textual 3-way merge produces on unstructured prose.

## Inputs (passed via $ARGUMENTS as a single argument block)

The caller writes three files to disk and passes their paths as arguments:

```
path:          relative path of the memory being merged (for context)
base_file:     /tmp/... — contents at the common ancestor commit
ours_file:     /tmp/... — contents on the local (target) branch tip
theirs_file:   /tmp/... — contents on the incoming branch tip
output_file:   /tmp/... — where to write the merged result
```

## Your task

1. Read `base_file`, `ours_file`, `theirs_file` with Read.
2. Produce a merged version that:
   - Preserves **every observation** present in ours or theirs (union, not intersection).
   - **Deduplicates** observations that appear in both in different wording — keep one, pick the clearer wording.
   - **Unifies sections with the same heading** — if both add entries under `## Current Patterns`, put them all under a single `## Current Patterns` section, not two adjacent sections.
   - Resolves contradictions by **keeping both with context** (e.g. "ours (machine A): X. theirs (machine B): Y."). Never silently drop a contradicting observation.
   - Preserves ordering from the more structured side when ordering matters.
   - Retains the base file's skeleton (top-level headings) unless both sides clearly reorganized it.
3. Write the merged content to `output_file` using Write.
4. Do NOT add commentary, explanations, headers like "## Merged Output", or metadata. The file contents must be valid memory content, ready to be read verbatim by the agent the memory belongs to.
5. Do NOT invent observations that aren't in any input. No hallucinated code details, file paths, or decisions.

## Constraints

- Markdown format preserved (headings, bullets, code fences).
- Frontmatter if present in inputs: carry over from `ours_file` (local branch wins on metadata).
- If `theirs_file` is empty or is just a placeholder like `(empty — last consolidated by Dreamer)`, prefer `ours_file` verbatim.
- If `ours_file` is empty/placeholder and `theirs_file` has content, prefer `theirs_file` verbatim.
- If base/ours/theirs are all identical, write any of them unchanged to `output_file`.

## After writing

Report a single line: `merged <path>: <n_observations_kept>` and nothing else. The caller parses this line.

## Additional arguments
$ARGUMENTS
