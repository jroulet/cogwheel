#!/usr/bin/env bash
# Mind-meld merge — one script with three modes.
#
# Usage:
#   bash scripts/mind_meld.sh <branch> [--git]      # merge: orchestrate a mind-meld
#                                                     merge (default when first
#                                                     arg is not a known mode).
#                                                     --git: plain 3-way, no LLM.
#   bash scripts/mind_meld.sh resolve               # internal: resolve soft-blacklist
#                                                     conflicts mid-merge. Called by
#                                                     `merge` on conflict; can also
#                                                     be run manually if the user
#                                                     interrupted a merge.
#   bash scripts/mind_meld.sh triage                # internal: post-merge LLM triage
#                                                     on auto-merged memory files.
#                                                     Called by .claude/hooks/post-merge
#                                                     when MIND_MELD=1 is set.
#
# User-facing surface is just `git mind-meld <branch>` (the alias installed
# by bootstrap --with-agentic points at this script; the subcommand dispatch
# is an implementation detail).
#
# All three modes share one core: _llm_merge_path <path> <base-sha> <ours-ref>
# <theirs-ref>. Merge just orchestrates git + delegates to resolve on conflict;
# triage runs after an already-committed merge to clean up messy auto-merges;
# resolve runs during a mid-merge state.
#
# Cost: ~$0.05-0.20 per memory file merged by Sonnet. Full mind-meld round
# of all agent memories: $1-2 worst case.
#
# Requires: `claude` CLI, `/merge-memory` slash command installed as a user
# skill (not shipped with this skill — see references/mind-meld.md for setup).
#
# Ported from gw_detection_ias scripts/mind_meld.sh (71d986c, 172cdd8) without
# skill-specific changes — the script is generic. The blacklist
# (.claude/memory-blacklist.txt) drives all project-specific behavior.

set -uo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || exit 1
cd "$REPO_ROOT"

BLACKLIST="$REPO_ROOT/.claude/memory-blacklist.txt"

# ── Shared helpers ──────────────────────────────────────────────────────

# Populate `soft_paths` array from the blacklist. Hard blacklist is never
# touched by LLM merge — those paths are per-machine state that shouldn't
# merge semantically anyway.
_load_soft_blacklist() {
    soft_paths=()
    [ -f "$BLACKLIST" ] || return 0
    local section=""
    while IFS= read -r line; do
        line="${line%%#*}"
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        [ -z "$line" ] && continue
        case "$line" in
            "[soft]") section="soft"; continue ;;
            "[hard]") section="hard"; continue ;;
        esac
        [ "$section" = "soft" ] && soft_paths+=("$line")
    done < "$BLACKLIST"
}

_is_soft_blacklisted() {
    local target="$1" path
    for path in "${soft_paths[@]}"; do
        case "$path" in
            */) [[ "$target" == ${path}* ]] && return 0 ;;
            *)  [ "$target" = "$path" ]     && return 0 ;;
        esac
    done
    return 1
}

# LLM-merge one path. Args: <relative_path> <base-sha> <ours-ref> <theirs-ref>.
# Extracts each version via `git show`, invokes /merge-memory, writes result
# back to working tree. Returns 0 on success, 1 on failure.
_llm_merge_path() {
    local path="$1" base="$2" ours="$3" theirs="$4"
    local tmp_root="$REPO_ROOT/.claude/handoff/mm_$$_$(echo "$path" | tr '/' '_')"
    mkdir -p "$tmp_root"

    local base_f="$tmp_root/base"
    local ours_f="$tmp_root/ours"
    local theirs_f="$tmp_root/theirs"
    local out_f="$tmp_root/merged"

    git show "$base:$path"   > "$base_f"   2>/dev/null || : > "$base_f"
    git show "$ours:$path"   > "$ours_f"   2>/dev/null || : > "$ours_f"
    git show "$theirs:$path" > "$theirs_f" 2>/dev/null || : > "$theirs_f"

    local args="path: $path
base_file: $base_f
ours_file: $ours_f
theirs_file: $theirs_f
output_file: $out_f"

    if ! claude -p "/merge-memory $args" \
            --model claude-sonnet-4-6 \
            --print \
            --dangerously-skip-permissions \
            >/dev/null 2>&1; then
        rm -rf "$tmp_root"
        return 1
    fi

    if [ ! -s "$out_f" ]; then
        rm -rf "$tmp_root"
        return 1
    fi

    cp -f "$out_f" "$REPO_ROOT/$path"
    rm -rf "$tmp_root"
    return 0
}

# Heuristic: does a file need LLM semantic merge after git auto-merged?
# Signals: conflict markers present, duplicate section headers, or both
# sides added >5 non-overlapping lines.
_needs_llm_merge() {
    local path="$1" abs="$REPO_ROOT/$path"
    [ -f "$abs" ] || return 1
    if grep -qE '^(<<<<<<< |=======|>>>>>>> )' "$abs" 2>/dev/null; then
        return 0
    fi
    local dup
    dup=$(grep -E '^## ' "$abs" 2>/dev/null | sort | uniq -d | head -1)
    [ -n "$dup" ] && return 0
    local ours_adds theirs_adds
    ours_adds=$(git diff --numstat HEAD^2 HEAD^1 -- "$path" 2>/dev/null | awk '{print $1}' | head -1)
    theirs_adds=$(git diff --numstat HEAD^1 HEAD^2 -- "$path" 2>/dev/null | awk '{print $1}' | head -1)
    [ "${ours_adds:-0}" -gt 5 ] && [ "${theirs_adds:-0}" -gt 5 ] && return 0
    return 1
}

# ── Mode: triage (post-merge) ────────────────────────────────────────────

cmd_triage() {
    # Only valid in a three-way merge commit (HEAD has two parents).
    git rev-parse HEAD^2 >/dev/null 2>&1 || return 0

    # Guarded by env: only run when mind-meld is active AND LLM wasn't opted out.
    [ "${MIND_MELD:-0}" = "1" ] || return 0
    [ "${MIND_MELD_LLM:-1}" = "1" ] || return 0

    command -v claude >/dev/null 2>&1 || {
        echo "  mind_meld triage: claude CLI not found — skipping."
        return 0
    }

    _load_soft_blacklist
    [ "${#soft_paths[@]}" -eq 0 ] && return 0

    local base_sha
    base_sha=$(git merge-base HEAD^1 HEAD^2 2>/dev/null || echo "")
    [ -z "$base_sha" ] && return 0

    local changed merged_any=0
    changed=$(git diff --name-only HEAD^1 HEAD 2>/dev/null || true)
    [ -z "$changed" ] && return 0

    while IFS= read -r path; do
        [ -z "$path" ] && continue
        _is_soft_blacklisted "$path" || continue
        _needs_llm_merge "$path" || continue
        echo "  mind_meld triage: merging $path"
        if _llm_merge_path "$path" "$base_sha" "HEAD^1" "HEAD^2"; then
            git add "$path" 2>/dev/null || true
            merged_any=1
        else
            echo "    LLM merge failed, keeping git auto-merge for $path"
        fi
    done <<< "$changed"

    if [ "$merged_any" -eq 1 ]; then
        git commit --amend --no-edit --no-verify >/dev/null 2>&1 || true
        echo "  mind_meld triage: amended merge commit with LLM-merged memories"
    fi
}

# ── Mode: resolve (mid-merge conflict) ───────────────────────────────────

cmd_resolve() {
    local commit_after=0
    [ "${1:-}" = "--commit" ] && commit_after=1

    [ -f "$(git rev-parse --git-dir)/MERGE_HEAD" ] || {
        echo "  mind_meld resolve: not in a merge state." >&2
        return 1
    }
    command -v claude >/dev/null 2>&1 || {
        echo "  mind_meld resolve: claude CLI not found." >&2
        return 1
    }

    _load_soft_blacklist
    [ "${#soft_paths[@]}" -eq 0 ] && return 0

    local conflicted
    conflicted=$(git diff --name-only --diff-filter=U 2>/dev/null)
    [ -z "$conflicted" ] && { echo "  mind_meld resolve: no conflicts."; return 0; }

    local base_sha
    base_sha=$(git merge-base HEAD MERGE_HEAD 2>/dev/null || echo "")
    [ -z "$base_sha" ] && { echo "  mind_meld resolve: no merge base." >&2; return 1; }

    local n_resolved=0
    while IFS= read -r path; do
        [ -z "$path" ] && continue
        _is_soft_blacklisted "$path" || continue
        echo "  mind_meld resolve: merging $path"
        if _llm_merge_path "$path" "$base_sha" "HEAD" "MERGE_HEAD"; then
            git add "$path"
            n_resolved=$((n_resolved + 1))
        else
            echo "    LLM merge failed, leaving conflict markers in $path"
        fi
    done <<< "$conflicted"

    if [ "$n_resolved" -eq 0 ]; then
        echo "  mind_meld resolve: no soft-blacklisted conflicts to merge."
        return 0
    fi

    echo "  mind_meld resolve: semantically merged $n_resolved file(s)."
    local remaining
    remaining=$(git diff --name-only --diff-filter=U 2>/dev/null)
    if [ -n "$remaining" ]; then
        echo "  Unresolved non-blacklisted conflicts remain:"
        echo "$remaining" | sed 's/^/    - /'
        echo "  Resolve manually, then: git commit --no-edit"
        return 0
    fi

    if [ "$commit_after" -eq 1 ]; then
        git commit --no-edit --no-verify
        echo "  mind_meld resolve: merge commit finalized."
    fi
}

# ── Mode: merge (default; user-facing) ───────────────────────────────────

cmd_merge() {
    local mode=llm
    local branch=""
    local extra=()

    while [ $# -gt 0 ]; do
        case "$1" in
            --git) mode=git; shift ;;
            --llm) mode=llm; shift ;;
            -*) extra+=("$1"); shift ;;
            *)
                if [ -z "$branch" ]; then
                    branch="$1"
                else
                    extra+=("$1")
                fi
                shift
                ;;
        esac
    done

    if [ -z "$branch" ]; then
        echo "Usage: git mind-meld [--llm|--git] <branch>" >&2
        return 1
    fi

    export MIND_MELD=1
    [ "$mode" = "git" ] && export MIND_MELD_LLM=0

    echo "mind-meld: starting $mode merge of $branch"

    set +e
    git merge "$branch" --no-commit --no-ff --no-edit "${extra[@]}"
    set -e

    local conflicts
    conflicts=$(git diff --name-only --diff-filter=U 2>/dev/null || true)

    if [ -z "$conflicts" ]; then
        # Clean auto-merge. Commit — post-merge hook will run triage.
        git commit --no-edit --no-verify
        echo "mind-meld: merge completed."
        return 0
    fi

    echo ""
    echo "mind-meld: conflicts in:"
    echo "$conflicts" | sed 's/^/  - /'

    if [ "$mode" = "git" ]; then
        cat <<MSG

mind-meld: --git mode — leaving conflicts for manual resolution.
  Resolve them in your editor, then:
    git commit --no-edit
MSG
        return 1
    fi

    cmd_resolve

    local remaining
    remaining=$(git diff --name-only --diff-filter=U 2>/dev/null || true)
    if [ -n "$remaining" ]; then
        cat <<MSG

mind-meld: $(echo "$remaining" | grep -c .) non-blacklisted conflict(s) remain:
$(echo "$remaining" | sed 's/^/  - /')

These are shared content (pipeline code, specs, fragments). Resolve manually.
When done:
  git commit --no-edit
MSG
        return 1
    fi

    git commit --no-edit --no-verify
    echo ""
    echo "mind-meld: merge completed with LLM-resolved memory conflicts."
}

# ── Entry point: dispatch by subcommand ─────────────────────────────────

case "${1:-}" in
    triage)  shift; cmd_triage "$@" ;;
    resolve) shift; cmd_resolve "$@" ;;
    merge)   shift; cmd_merge "$@" ;;
    -h|--help)
        cat <<HELP
Usage:
  git mind-meld [--llm|--git] <branch>     # default: LLM semantic merge
  bash scripts/mind_meld.sh triage         # internal (post-merge hook)
  bash scripts/mind_meld.sh resolve        # internal (mid-merge recovery)

User-facing command is \`git mind-meld\`. Triage and resolve are internal.
HELP
        ;;
    *)
        # No subcommand given — treat first arg as branch for merge.
        cmd_merge "$@"
        ;;
esac
