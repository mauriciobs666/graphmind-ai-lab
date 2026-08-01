# `kiro-demo-agent` — Test Plan

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** — (—)

## 1. Scope & objective

Live acceptance pass for the minimal Kiro CLI demo agent (`kiro/.kiro/agents/falkor-chat-demo.json`)
that connects to `falkor-chat`'s MCP server as a client. The static half (schema validation,
`kiro-cli agent list`) is already done and out of scope here — this plan covers the **live** half:
actually driving `kiro-cli` against a running `falkor-chat` and observing real behavior, against
AC-1…AC-4 from `kiro/docs/requirements/kiro-demo-agent.md`.

References:
- Requirements: `kiro/docs/requirements/kiro-demo-agent.md` (AC-1…AC-4, FR-1…FR-6, out-of-scope list).
- Plan: `kiro/docs/plans/kiro-demo-agent.md`, especially §5 ("Test / verification strategy") — this
  plan follows that recipe rather than re-deriving one.
- Config under test: `kiro/.kiro/agents/falkor-chat-demo.json`.
- Quick-start: `kiro/README.md`.
- falkor-chat MCP surface (unchanged ground truth): `falkor-chat/docs/DESIGN.md` §15,
  `falkor-chat/server/falkorchat/mcp.py`.

## 2. Risk assessment

- **Highest risk — the two-way live round trip (AC-1/AC-2) has never been run.** Everything to
  date (plan §2.3) was verified with the CLI pointed at scratch directories or an unreachable
  `localhost:8000`; nobody has driven a real `send_message`/`read_messages` call through a live
  falkor-chat + LLM-backed `assistant` responder. This is the core demo promise — prioritize it.
- **Tool-surface containment (AC-4) is the security/scope-creep risk.** If `tools` accidentally
  admitted a bare `@falkor-chat` or `"*"`, the demo agent would silently gain
  `create_thread`/`create_channel`/`list_channels`/`list_threads`/`search_messages` and built-ins —
  a real containment failure, not cosmetic. Check both statically (file content) and live
  (`/tools` in-session), per the plan's own two-pronged AC-4 recipe.
- **The interactive `@`-file-completion risk is explicitly flagged as unverified by the plan**
  (§2.3, §5, §7) — only the non-interactive path was checked before now. This is a real,
  named residual risk, not a hypothetical one invented by this plan — worth a dedicated,
  genuinely-interactive test item (not `--no-interactive`, which sidesteps the very mechanism
  being tested).
- **Cold-start failure mode (falkor-chat not up yet) is a documented behavior claim**
  (`--require-mcp-startup` "fails loudly ... exit code 3") that has not been empirically confirmed
  for this specific agent config — cheap to check, worth confirming rather than trusting the flag's
  general docs blindly.
- **AC-3's "fresh clone" is expensive to do fully faithfully** (a second falkor-chat instance is out
  of proportion for this pass) — adopt the plan's own named adaptation: a real `git clone` to a
  scratch path (cheap, no second falkor-chat needed since `kiro-cli agent list` and the live chat
  flow both only need the *config* to be freshly sourced, not a second server) plus a repo-tree
  check for stray global Kiro config that would let a real fresh-clone user unknowingly rely on
  something not checked in.
- **Lower risk, still worth a pass**: the "no mention" edge case (mentions is optional — should not
  break `send_message`) is low-risk mechanically (single optional arg) but is the one edge case the
  plan explicitly calls out as worth checking that a plain post still works.

**Explicitly not (re-)tested here:**
- Static config schema validity / `kiro-cli agent list` scope — already run and passing per the
  dispatching brief; re-run only incidentally as a pre-flight sanity check, not as a scored item.
- falkor-chat's own MCP tool implementations (`send_message`/`read_messages` correctness,
  mention-triggered responder logic) — covered by falkor-chat's own test suite and prior QA passes
  (`falkor-chat/docs/test-reports/mention-reply-delivery-report.md`); this pass treats that path as
  already-verified plumbing and focuses on the Kiro-side client behavior.
- LLM response *quality/content* of `assistant`'s replies — out of scope; AC-2 only requires the
  reply text to be surfaced back in the Kiro session, not that its content is good.
- A true second `falkor-chat` instance for AC-3 — adapted per above, noted as a deliberate scope
  reduction, not an oversight.

## 3. Test items

| ID | Title | Type | Priority |
|---|---|---|---|
| TP-001 | AC-1: non-interactive send with mention lands in falkor-chat | e2e | High |
| TP-002 | AC-1 edge: literal `@assistant` typed in the true interactive TUI | exploratory | Medium |
| TP-003 | AC-2: read messages back surfaces `assistant`'s reply in the Kiro session | e2e | High |
| TP-004 | AC-3: fresh clone needs no manual wiring beyond checked-in config | integration | High |
| TP-005 | AC-4 (static): config file's `tools`/`allowedTools` are exactly the two entries | contract | Medium |
| TP-006 | AC-4 (live): `/tools` in-session shows exactly `send_message` + `read_messages` | contract | High |
| TP-007 | Edge: message without a mention still posts (no responder trigger) | functional | Low |
| TP-008 | Edge: cold start before falkor-chat is up fails loudly, not silently | resilience | Medium |

### TP-001 — AC-1: non-interactive send with mention

- **Preconditions:** falkor-chat up (`curl http://localhost:8000/` → 200); CWD = `kiro/`.
- **Steps:** Record current message count in `demo-welcome` via REST
  (`GET /threads/demo-welcome/messages?ws=acme`). Run
  `kiro-cli chat --agent falkor-chat-demo --require-mcp-startup --no-interactive "post 'hello from the kiro demo' and mention assistant"`.
  Capture stdout/tool-call trace. Re-query the REST endpoint.
- **Expected:** CLI exits 0; its trace shows `send_message` invoked (not another tool) with
  `re: "demo-welcome"`, `mentions` including `assistant`; the REST message list grows by exactly
  one new `user`-role message authored by `u1`/"Demo User" (per FR-6 — not a "Kiro" identity) whose
  text matches what was posted.
- **Priority:** High.

### TP-002 — AC-1 edge: literal `@assistant` in the true interactive TUI

- **Preconditions:** Same as TP-001. Requires a genuinely interactive session (e.g. driven via
  `tmux`/`script`, not `--no-interactive`, which the plan itself notes sidesteps this exact risk).
- **Steps:** Start `kiro-cli chat --agent falkor-chat-demo --require-mcp-startup` interactively.
  Type a message containing the literal substring `@assistant` (not paraphrased) and press Enter
  without Tab-selecting any file-completion suggestion that may pop up.
- **Expected:** No client-side file-completion swallows or alters the typed text; the message
  reaches the model call with `@assistant` intact (or the agent still correctly calls
  `send_message` with the intended content/mention) — the plan (§2.3/§7) flags this as an
  unverified, only-partially-mitigated risk; this item exists to close that gap.
- **Priority:** Medium (named risk in the plan, not a hard AC).

### TP-003 — AC-2: read the reply back

- **Preconditions:** TP-001 has completed and `assistant` has had time to reply (LLM-backed,
  asynchronous — allow several seconds; sanity-check via REST/web UI first).
- **Steps:** In the same or a fresh `kiro-cli` session against `falkor-chat-demo`, send
  "read the latest messages" (non-interactive is acceptable here since this item is not testing
  the `@`-completion risk).
- **Expected:** CLI trace shows `read_messages` invoked with `re: "demo-welcome"`; the reply text
  authored by `assistant` (matching what's visible in the REST/web UI) appears in the Kiro
  session's own output, not just silently fetched.
- **Priority:** High.

### TP-004 — AC-3: fresh clone, no manual wiring

- **Preconditions:** Scratch directory available; network access to the repo's local `.git`.
- **Steps:** `git clone` this repository to a scratch path. `cd <clone>/kiro`. Run
  `kiro-cli agent list`. Then repeat a minimal TP-001/TP-003-style round trip from that clone
  (`--no-interactive`, non-interactive send + read) against the same running falkor-chat instance
  (real falkor-chat re-use, per this plan's §2 adaptation — not a second server). Separately,
  confirm from the real repo tree that `~/.kiro/agents/` holds only the pre-existing
  `agent_config.json.example` (no stray personal agent) and that no `~/.kiro/mcp.json` /
  workspace `mcp.json` exists that the checked-in config secretly depends on.
- **Expected:** `falkor-chat-demo` is listed (`Workspace` scope) immediately from the clone, no
  extra setup step beyond `cd kiro`; the round trip succeeds identically to TP-001/TP-003; no
  stray global Kiro config is found.
- **Priority:** High.

### TP-005 — AC-4 (static): config file tool arrays

- **Steps:** Read `kiro/.kiro/agents/falkor-chat-demo.json` directly; inspect `tools` and
  `allowedTools`.
- **Expected:** Both arrays contain exactly `"@falkor-chat/send_message"` and
  `"@falkor-chat/read_messages"` — no bare `"@falkor-chat"`, no `"*"`, no built-ins.
- **Priority:** Medium.

### TP-006 — AC-4 (live): `/tools` in-session

- **Preconditions:** falkor-chat up.
- **Steps:** Inside an interactive `kiro-cli chat --agent falkor-chat-demo` session, run `/tools`.
- **Expected:** The listed/reachable tool set is exactly `send_message` and `read_messages` from
  the `falkor-chat` server — no `create_thread`/`create_channel`/`list_channels`/`list_threads`/
  `search_messages`, no built-in tools (`fs_read`, `execute_bash`, etc.).
- **Priority:** High.

### TP-007 — Edge: message without a mention

- **Steps:** `kiro-cli chat --agent falkor-chat-demo --no-interactive "post 'plain post, no mention' without mentioning assistant"`.
- **Expected:** `send_message` still succeeds (message lands in `demo-welcome`); `assistant` does
  not reply to this specific message (no responder trigger).
- **Priority:** Low.

### TP-008 — Edge: cold start before falkor-chat is up

- **Preconditions:** falkor-chat **not** running (stop it first if needed for this one item, then
  restart afterward — see report for final state).
- **Steps:** `cd kiro && kiro-cli chat --agent falkor-chat-demo --require-mcp-startup --no-interactive "hello"`.
- **Expected:** Fails fast with a clear, non-zero exit (documented as exit code 3 for
  `--require-mcp-startup`) rather than silently proceeding without the MCP tools.
- **Priority:** Medium.

## 4. Environment & data setup

- falkor-chat started via `cd falkor-chat && ./scripts/start_server.sh` (bootstraps FalkorDB,
  schema, seeds `assistant`/`demo-general`/`demo-welcome`, starts uvicorn with the AI responder
  enabled). Confirmed reachable via `curl http://localhost:8000/` → 200 before any test item runs.
- `demo-welcome` already contains pre-existing messages from prior QA passes (32 messages at the
  start of this run) — test items compare message **counts/deltas** and specific new content, not
  an assumption of an empty thread.
- `kiro-cli` v2.14.1 at `~/.local/bin/kiro-cli`, already authenticated on this machine (assumed
  per the dispatching brief; not re-verified via `kiro-cli doctor` in this pass — see report for
  why).
- All commands run from `kiro/` (CWD is load-bearing per the plan's exact-CWD agent-discovery
  finding, §2.3).

## 5. Entry / exit criteria

- **Entry:** falkor-chat reachable at `http://localhost:8000/`; `kiro-cli agent validate` and
  `kiro-cli agent list` (from `kiro/`) both already pass (confirmed pre-existing, re-checked as a
  cheap sanity step, not scored).
- **Exit:** All eight test items executed with a recorded pass/fail/blocked outcome and evidence;
  every failure written up as a defect in the report; falkor-chat's final state (left running or
  stopped) recorded.

## 6. Out of scope

- A true second falkor-chat instance for AC-3 (adapted — see §2/TP-004).
- falkor-chat's own MCP tool correctness and the LLM responder's reply quality (already covered
  elsewhere, see §2).
- `kiro-cli doctor`'s auto-remediation behavior — noted as an incidental finding if encountered,
  not a scored test item (it is not part of this feature).
- Multi-agent coordination, turn-taking, or artifact provenance (explicitly out of scope in the
  requirements themselves).
