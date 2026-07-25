# Kaizen — Learnings Inbox: devops

> Append-only capture of durable, non-obvious environment facts the `devops` agent
> discovers during runs — raw observations, not conclusions. The maintainer (cobb)
> periodically distills this inbox (agent-maintenance skill §5): verifies each entry,
> routes it (prompt / knowledge base / project docs / discard), logs the promotion in
> `history.md`, and clears it. The agent only appends here; it never promotes.
>
> Entry format (append at the end):
>
> ```markdown
> ## YYYY-MM-DD — <the fact, one line>
> - **Evidence:** what was run/read/observed (command, file:line, output)
> - **Context:** the task where it surfaced, one line
> - **Suggested home:** prompt | knowledge base | project docs | unsure
> ```

## 2026-07-19 — `joern-parse --version` is not a valid flag (throws a Scala stack trace); get the version from `joern --version`
- **Evidence:** `joern-parse --version` → `io.joern.joerncli.JoernParse$.parseConfig` stack trace (rc still 0 via pipe). `joern --version` prints `Version: 4.0.579` then drops into the REPL prompt. `joern-parse --help` works and lists no `--version`.
- **Context:** smoke-testing a fresh Joern v4.0.579 install; the SKILL suggested `--version` or `--help` for the smoke test.
- **Suggested home:** project docs (skills/joern-cpg/SKILL.md smoke-test note) — prefer `joern --version` for a version check.

## 2026-07-19 — Joern's Python frontend (`pysrc2cpg`) is bundled in the joern-cli zip; no cold-start runtime download
- **Evidence:** immediately after unzip, `joern-parse --language pythonsrc` on a 1-file sample succeeded in ~5s (rc 0, overlays applied, valid cpg.bin), invoking `/home/mauricio/joern/joern-cli/pysrc2cpg` in a separate process. No network fetch, no first-run stall. `--list-languages` shows `pythonsrc` and `python`.
- **Context:** task asked to pre-warm the Python frontend so a later parse isn't stalled on a cold download — turned out there is no such download; the frontend ships in the distribution.
- **Suggested home:** knowledge base (Joern install ops).

## 2026-07-19 — Joern release `.sha512` sidecar carries a build-relative path (`target/joern-cli.zip`), not the local filename
- **Evidence:** `$HOME/joern/joern-cli.zip.sha512` content = `<hash>  target/joern-cli.zip`; `sha512sum -c` would fail on the path. Compare the hash column only (`awk '{print $1}'`).
- **Context:** verifying the downloaded Joern distribution before unzip.
- **Suggested home:** knowledge base (Joern install ops).

## 2026-07-25 — Claude Code MCP: `.mcp.json` discovery walks up to the repo root, but project-approval scope is keyed on the session's cwd
- **Evidence:** with `.mcp.json` + `"enabledMcpjsonServers": ["cpg"]` at the repo root, `claude mcp list` from the repo root → `cpg: … - ✔ Connected`; the same command from the `falkor-chat/` subdirectory → `⏸ Pending approval (run \`claude\` to approve)`. The server was still *discovered* from the subdir (walk-up to the git root worked), so this is approval scoping, not path resolution. `~/.claude.json` `projects` has exactly one entry for this repo (`<repo-root>`, `hasTrustDialogAccepted: true`) and none for the subdirectory, and `falkor-chat/` carries its own `.claude/settings.local.json` — so a subdir session is a *separate* settings/approval scope and the root's pre-approval does not reach it.
- **Context:** wiring the repo's first MCP server (`cpg`) at project scope; the plan's done-condition included "works from a session started in a subdirectory", which turns out to need one extra interactive approval per subdirectory.
- **Suggested home:** knowledge base / `skills/agent-standards/claude-code.md` §MCP — the scope table should say that `enabledMcpjsonServers` pre-approval is cwd-scoped even though `.mcp.json` discovery is root-scoped.

## 2026-07-25 — An MCP server's `.mcp.json` launch shape can be verified end-to-end from a plain shell, without restarting the harness
- **Evidence:** reading `.mcp.json`, spawning `command` + `args` verbatim with `CLAUDE_PROJECT_DIR` injected into the child env (which is exactly where Claude Code sets it) and `cwd=/tmp`, then speaking raw JSON-RPC (`initialize` → `tools/list` → `tools/call`) over the pipes reproduced everything the in-session `/mcp` view would show: `serverInfo {'name': 'cpg'}`, tool count 1, `required: ['graph','cypher']`, `outputSchema: None`, `annotations {'readOnlyHint': True}`, `_meta {'anthropic/maxResultSizeChars': 60000}`, and a real query returning `rows=1 · count(m)=1968`. Launching from `/tmp` also proves the `bash -c "exec \"$CLAUDE_PROJECT_DIR/…\""` form is cwd-independent.
- **Context:** S3 of the CPG query-access plan — the harness-side done-conditions need a session restart, but the server-side contract does not, so the pending surface can be reduced to just the approval prompt.
- **Suggested home:** prompt (devops verification habit) or knowledge base — "verify the config's own command line, not just the script it points at".
