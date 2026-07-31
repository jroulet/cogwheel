/**
 * Cogwheel OpenCode Plugin — lifecycle hooks for the shared agent pipeline.
 *
 * Ports the functionality from .codex/hooks/ and .claude/hooks/:
 * 1. Use-serena enforcement: deny native Read/Edit/Write/Grep/Glob/Bash for
 *    project files, forcing the model to use mcp__serena__* equivalents
 * 2. Conda python routing: wrap bare python/pip commands in conda run
 * 3. Serena readiness gate: block native tools until Serena is initialized
 * 4. Session start context injection
 *
 * Install: auto-discovered from .opencode/plugins/
 */

import { execSync } from "child_process";
import { existsSync, readFileSync } from "fs";
import { resolve, relative } from "path";

// Track whether Serena has been initialized in this session
let serenaReady = false;
let projectRoot = "";
let _condaEnv: string | null = null;

function getCondaEnv(): string {
  if (_condaEnv !== null) return _condaEnv;
  // Priority: process.env.SDK_CONDA_ENV (if set by user), then .env file,
  // then fallback. Never hardcode a machine-specific default.
  const fromEnv = process.env.SDK_CONDA_ENV;
  if (fromEnv && fromEnv !== "cogwheel_310") {
    _condaEnv = fromEnv;
    return _condaEnv;
  }
  try {
    const root = getProjectRoot();
    const envFile = readFileSync(resolve(root, ".env"), "utf-8");
    const match = envFile.match(/^SDK_CONDA_ENV=["']?([^"'\n]+)["']?/m);
    if (match) {
      _condaEnv = match[1];
      return _condaEnv;
    }
  } catch {}
  _condaEnv = fromEnv || "cogwheel_310";
  return _condaEnv;
}

function getProjectRoot() {
  if (!projectRoot) {
    try {
      projectRoot = execSync("git rev-parse --show-toplevel", {
        encoding: "utf-8",
      }).trim();
    } catch {
      projectRoot = process.cwd();
    }
  }
  return projectRoot;
}

function isProjectFile(filePath: string): boolean {
  if (!filePath) return false;
  const root = getProjectRoot();
  const abs = filePath.startsWith("/") ? filePath : resolve(root, filePath);
  // Not a project file if outside root or inside .claude/
  if (!abs.startsWith(root + "/")) return false;
  if (abs.startsWith(root + "/.claude/")) return false;
  if (abs.startsWith(root + "/.opencode/")) return false;
  return true;
}

function isGitignored(filePath: string): boolean {
  try {
    const root = getProjectRoot();
    const rel = relative(root, filePath.startsWith("/") ? filePath : resolve(root, filePath));
    execSync(`git -C "${root}" check-ignore -q "${rel}"`, { encoding: "utf-8" });
    return true;
  } catch {
    return false;
  }
}

function isImageOrPdf(filePath: string): boolean {
  return /\.(png|jpg|jpeg|gif|svg|pdf|ipynb)$/i.test(filePath);
}

// Allowlisted bash commands that pass through without Serena redirection
const BASH_PASSTHROUGH = /^(git|gh|conda|brew|npm|npx|which|chmod|mkdir|ls|stat|wc|pwd|date|env|printenv|df|du|file|ps|pgrep|diff|kill|pkill)(\s|$)/;
const PROJECT_SCRIPT = /^\.(claude\/(sdk|hooks)\/[A-Za-z0-9_.-]+\.sh|codex\/build|codex\/hooks\/[A-Za-z0-9_.-]+\.sh|opencode\/build|opencode\/resume_driver\.sh)(\s|$)/;

export default (async ({ client, project, directory, $ }) => {
  projectRoot = directory || "";

  return {
    "permission.ask": async (input, output) => {
      const permType = input.type;
      const pattern = Array.isArray(input.pattern)
        ? input.pattern[0] || ""
        : input.pattern || "";
      const metadata = input.metadata || {};

      // ── Serena readiness gate (build mode only) ──
      if (
        process.env.AGENT_PROVIDER === "opencode" &&
        process.env.OPENCODE_SERENA_URL &&
        !serenaReady
      ) {
        if (["read", "edit", "glob", "grep", "bash"].includes(permType)) {
          output.status = "deny";
          return;
        }
      }

      // ── Use-serena enforcement for project files ──
      if (permType === "read") {
        const fp = pattern;
        if (isImageOrPdf(fp) || isGitignored(fp) || !isProjectFile(fp)) {
          output.status = "allow";
          return;
        }
        output.status = "deny";
        return;
      }

      if (permType === "edit") {
        const fp = pattern;
        if (isGitignored(fp) || !isProjectFile(fp)) {
          output.status = "allow";
          return;
        }
        output.status = "deny";
        return;
      }

      if (permType === "glob" || permType === "grep") {
        // If path targets project, deny and redirect to Serena
        if (!pattern || isProjectFile(pattern)) {
          output.status = "deny";
          return;
        }
        output.status = "allow";
        return;
      }

      if (permType === "bash") {
        const command = pattern || (metadata as any).command || "";
        // Strip leading VAR=value assignments
        let stripped = command.replace(
          /^([A-Za-z_][A-Za-z0-9_]*=[^\s]*\s+)+/,
          ""
        );
        // Allow passthrough commands
        if (BASH_PASSTHROUGH.test(stripped)) {
          output.status = "allow";
          return;
        }
        // Allow project scripts
        if (PROJECT_SCRIPT.test(stripped)) {
          output.status = "allow";
          return;
        }
        // Deny general bash — redirect to Serena execute_shell_command
        output.status = "deny";
        return;
      }
    },

    "tool.execute.before": async (input, output) => {
      const toolName = input.tool;
      const toolArgs = output.args || {};

      // ── Conda python routing for bash ──
      if (toolName === "bash") {
        const command = toolArgs.command || "";
        if (
          /(?:^|\s)(python3?|pip3?)(?:\s|$)/.test(command) &&
          !/(?:^|\s)conda(?:\s+run)?(?:\s|$)/.test(command)
        ) {
          const envName = getCondaEnv();
          output.args = {
            ...toolArgs,
            command: `conda run -n ${envName} ${command}`,
          };
        }
        return;
      }

      // ── Conda python routing for Serena execute_shell_command ──
      if (
        toolName === "mcp__serena__execute_shell_command" ||
        toolName === "mcp__serena_build__execute_shell_command"
      ) {
        const command = toolArgs.command || "";
        if (
          /(?:^|\s)(python3?|pip3?)(?:\s|$)/.test(command) &&
          !/(?:^|\s)conda(?:\s+run)?(?:\s|$)/.test(command)
        ) {
          const envName = getCondaEnv();
          output.args = {
            ...toolArgs,
            command: `conda run -n ${envName} ${command}`,
          };
        }
        return;
      }
    },

    "tool.execute.after": async (input, output) => {
      const toolName = input.tool;
      const toolArgs = input.args || {};

      // Mark Serena ready after successful initial_instructions
      if (
        toolName === "mcp__serena__initial_instructions" ||
        toolName === "mcp__serena_build__initial_instructions"
      ) {
        serenaReady = true;
      }

      // ── Professor auto-mark-read ──
      // After write_memory to professor/* topics, extract arxiv IDs and
      // create read markers at .serena/memories/professor/read.d/<id>
      if (
        toolName === "mcp__serena__write_memory" ||
        toolName === "mcp__serena_build__write_memory"
      ) {
        const memoryName = toolArgs.memory_name || "";
        const content = toolArgs.content || "";

        // Only fire on professor topic memory writes (not read.d markers)
        if (
          memoryName.startsWith("professor/") &&
          !memoryName.startsWith("professor/read.d/") &&
          content
        ) {
          try {
            const root = getProjectRoot();
            const readDir = resolve(root, ".serena/memories/professor/read.d");
            const { mkdirSync, writeFileSync } = await import("fs");
            mkdirSync(readDir, { recursive: true });

            // Extract arxiv IDs (YYYY.NNNNN format)
            const ids = content.match(/\d{4}\.\d{4,5}/g);
            if (ids) {
              const unique = [...new Set(ids)];
              for (const id of unique) {
                const marker = resolve(readDir, id);
                if (!existsSync(marker)) {
                  writeFileSync(marker, "", { flag: "wx" });
                }
              }
            }
          } catch {
            // Silent — hooks shouldn't be chatty
          }
        }
      }
    },

    "shell.env": (input, output) => {
      // Inject AGENT_PROVIDER into shell environment
      if (process.env.AGENT_PROVIDER) {
        output.env.AGENT_PROVIDER = process.env.AGENT_PROVIDER;
      }
      // Do NOT set SDK_CONDA_ENV here — the build scripts read it from
      // .env themselves. Setting a hardcoded default here pollutes child
      // processes with the wrong env name when .env says otherwise.
    },
  };
}) satisfies any;
