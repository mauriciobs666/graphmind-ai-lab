# Security review: P1-11b `.mcp.json` exec-path launch fact

> **Status:** active · **Owner:** `security-expert` · **Tracks:** P1-11b (C-1001)

**Lens:** code/app security. **Reviewed:** uncommitted `cpg/extractors/common/writer_rules.py` (`mcp_exec_launch_path`), `cpg/extractors/config/launch.py` (`mcp_drafts(exec_launch=)`), `cpg/extractors/config/extract.py`, `p1_11b` tests (155 pass), against `docs/plans/connected-code-graph-p1-extraction.md` P1-11b, `bundle.gate_argv`/`screen_fails`, and `ccg/linker/bind.py` `_script_of`.

**Verdict: approve with suggestions.** No blocker, no major. All probes ran in `cpg/`, `ulimit -v 1500000`, `timeout -s KILL 45`, script `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/a56a19ba-3e84-44da-9d76-47eef6534cc4/scratchpad/sec-p1-11b/probe.py`. No exploitation, no live targets.

CPG: considered, not relevant — a ~30-line pure function plus one call site; the path was traced by reading the code and probing it directly.

## Findings

### Minor 1 — a low-entropy or name-rule-word secret in the path is emitted (confirmed, accepted-class risk)
The path faces only the token screen (`screen_fails`, T1-T6) and the final `_plain_ok` (length ≤128, `_VALUE_CHARS`). The name rule (`name_fails`: `password`, `token`, `key`...) is not applied, since it only guards `--name value` pairs. Confirmed emitted: `scripts/hunter2-password.sh`, `scripts/token/run.sh`, `scripts/deadbeef012.sh` (hex run 11 < 12), `scripts/abc123def456xyz.sh` (15 < 16), `aGVsbG8/d29ybGQ/xyz.sh` (base64 split by `/`), `x/ab-cd-ef-...-st.sh` (dashed groups, T6 not hit), `.env`, `.ssh/id_rsa`, `a/.../b.sh`, `Scripts/RUN.sh`. A path segment short enough to dodge T1-T3 can still carry a short secret, but a repo path is already an at-rest bundle value elsewhere (`symbol.path`, mount paths), and the executed shell never needs it to be secret. Residual exposure is bounded by 11-15-char runs. Suggestion (optional, cheap): also reject a segment that `bundle.name_fails` flags, or one that starts with `.` beyond a single leading dot, if the owner wants the stricter reading of "never carries secrets". Not required to ship.

### Minor 2 — plan/code divergence: `${CLAUDE_PROJECT_DIR}` form
The plan's rule admits `\${CLAUDE_PROJECT_DIR}/`; the code admits only `$CLAUDE_PROJECT_DIR/` (probe: `exec "${CLAUDE_PROJECT_DIR}/a.sh"` -> `None`, stays redacted). Safe direction (stricter), but the plan text or code should be reconciled so a future reader does not "fix" it into a loosening. `bind.py` `_PROJECT_DIR` strips both forms, so the reader is already a superset.

### Nit 1 — address-shaped path accepted as a "path"
`10.0.0.1/x` and `192.168.1.1` pass and are emitted as `command`. Harmless (gate's address check ran and passed), but `_script_of` requires `/` or `.sh`; `192.168.1.1` has neither so is skipped. No action needed beyond awareness.

### Nit 2 — str/list subclasses accepted
A `str` or `list` subclass for command/args is accepted (probe: `S("bash")`, `L([...])` -> path). Output is a plain `str` (regex group). Inputs come from `launch._scalar_list`/`Node.text`, always plain, so this is theoretical. A tuple is correctly refused.

## Probed and clean (confirmed)
- **Regex:** one ASCII class, no nested quantifier, `fullmatch`, 600-char pre-cap; hostile 590-char inputs return in ~0.0000s. Trailing `\n` refused, NUL, tab, double space, zero-width, fullwidth and Arabic-Indic digits refused (class is ASCII literals, not `\d`/`\w`). Mismatched quotes refused.
- **Path hostility:** `..`, `.`, empty segment (`a//b`), absolute, trailing slash, leading `-`, backslash, `~` all refused. Remaining `...`/`....`/trailing-dot/case segments are ordinary names to `posixpath.normpath`, and `bind.py` joins lexically against the system dir with containment checks, without open/stat/resolve (its trust rule (a)), so a symlink-ish name cannot steer anything. A hostile `.mcp.json` can only name an in-repo path, the same power it already had via any `command` fact.
- **Secrets:** `sk-`, `ghp_` (T5), 12+ hex (T3), UUID (T4), 16+ mixed (T1), 24+ (T2), and paths >128 (gate `_plain_ok`) are all dropped; the emitted value is a fixed point of `gate_argv` (`launch_argv_fields([path])` returned unchanged in every accepted case).
- **Variables:** `$HOME`, `${...}`, `$(..)`, backticks, `;`, spaces, globs do not match the class, so cannot slip in.
- **Type confusion:** bytes, dict/list/None command, None args, non-str elements refused without raising.
- **Scope:** `exec_launch` is true only for `mcp-json`; Kiro agents are excluded (verified in `extract.py` and by tests).

## What's solid
Whole-string anchored, linear, length-capped allow-list; exact `["-c", S]` and exact-shell check; double screen (whole path and per segment); defence in depth through the unchanged `launch_argv_fields`/`gate_argv` fixed-point validator; thorough negative tests.

## Open questions
- Does the owner want Minor 1's stricter segment rule, or is "same as other repo paths in bundles" the accepted bar? (`architect`/`cobb` call; advisory only.)
