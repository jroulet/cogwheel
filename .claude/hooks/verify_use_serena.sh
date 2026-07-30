#!/usr/bin/env bash
# Behavioural regression check for the patched use-serena.sh (F046).
# deny() prints JSON and STILL exits 0, so allow/deny must be read from
# stdout, never from the exit code -- the mistake in my first probe.
cd /home/tejaswi/Work/cogwheel-claude-dev || exit 1
H=.claude/hooks/use-serena.sh
fail=0

check() {
    local payload="$1" expect="$2" label="$3" out rc verdict
    out=$(timeout 5 bash "$H" <<< "$payload" 2>&1); rc=$?
    if [ "$rc" -eq 124 ]; then
        verdict=HANG
    elif printf '%s' "$out" | grep -q '"permissionDecision"'; then
        verdict=DENY
    else
        verdict=ALLOW
    fi
    if [ "$verdict" = "$expect" ]; then
        printf '  ok    %-6s %s\n' "$verdict" "$label"
    else
        printf '  FAIL  got=%-6s want=%-6s %s\n' "$verdict" "$expect" "$label"
        fail=1
    fi
}

echo "=== F046 triggers (all HUNG before the patch) ==="
check '{"tool_name":"Bash","tool_input":{"command":"echo $(pgrep -af \"[s]dk/build.py\")"}}' DENY  'bracket class inside $( )'
check '{"tool_name":"Bash","tool_input":{"command":"echo $(grep [0-9] f)"}}'                DENY  'bracket range inside $( )'
check '{"tool_name":"Bash","tool_input":{"command":"echo $(ls [ab])"}}'                     DENY  'bracket class, allowlisted inner cmd'
check '{"tool_name":"Bash","tool_input":{"command":"ls $(echo `ls [xy]`)"}}'                ALLOW 'brackets in BACKTICKS inside $( )'

echo
echo "=== allow/deny behaviour must be UNCHANGED ==="
check '{"tool_name":"Bash","tool_input":{"command":"git status --short"}}'                  ALLOW 'git'
check '{"tool_name":"Bash","tool_input":{"command":"ls -l"}}'                               ALLOW 'ls'
check '{"tool_name":"Bash","tool_input":{"command":"pgrep -f pytest"}}'                     ALLOW 'pgrep'
check '{"tool_name":"Bash","tool_input":{"command":"LOG=/tmp/x.log git log -1"}}'           ALLOW 'VAR= prefix then git'
check '{"tool_name":"Bash","tool_input":{"command":"LOG=/tmp/b_$(date +%s).log git log -1"}}' ALLOW 'VAR= with $( ) value then git'
check '{"tool_name":"Bash","tool_input":{"command":".claude/sdk/launch_build.sh a b"}}'     ALLOW 'sdk script'
check '{"tool_name":"Bash","tool_input":{"command":".claude/hooks/install_hooks.sh"}}'      ALLOW 'hooks script'
check '{"tool_name":"Bash","tool_input":{"command":"cat foo.py"}}'                          DENY  'cat -> serena read_file'
check '{"tool_name":"Bash","tool_input":{"command":"grep -r x cogwheel/"}}'                 DENY  'grep -> serena search'
check '{"tool_name":"Bash","tool_input":{"command":"python -c pass"}}'                      DENY  'bare python'
check '{"tool_name":"Read","tool_input":{"file_path":"cogwheel/lensing/likelihood.py"}}'    DENY  'Read project file'
check '{"tool_name":"Read","tool_input":{"file_path":"/home/tejaswi/Work/cogwheel-claude-dev/.claude/spec/SPEC.md"}}' ALLOW '.claude path exempt'
check '{"tool_name":"Read","tool_input":{"file_path":"/tmp/elsewhere.py"}}'                 ALLOW 'non-project path'
check '{"tool_name":"Edit","tool_input":{"file_path":"cogwheel/lensing/likelihood.py"}}'    DENY  'Edit project file'
check '{"tool_name":"Write","tool_input":{"file_path":"cogwheel/new.py"}}'                  DENY  'Write project file'
check '{"tool_name":"Grep","tool_input":{"path":""}}'                                       DENY  'Grep'
check '{"tool_name":"Glob","tool_input":{"path":""}}'                                       DENY  'Glob'

echo
if [ "$fail" -eq 0 ]; then echo "HOOK VERIFY: ALL PASS"; else echo "HOOK VERIFY: FAILURES ABOVE"; fi
exit $fail
