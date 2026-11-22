# Anthropic API adoption: security review of the plan

> **Status:** active · **Owner:** `security-expert` · **Tracks:** — · **Reviews:** `docs/plans/anthropic-api-adoption.md`

## 1. Scope & verdict

Lenses: secrets/infra hardening (primary) and code/app security of what the plan will introduce. Reviewed the plan (§3.1, §3.3-§3.6, U1, U2, U4, U8, U10) against `docs/requirements/anthropic-api-adoption.md` (FR-10, FR-11, FR-14, FR-15, AC-7, AC-10, AC-12), and against the real code and config the plan cites. The analyst's concurrent review is in `docs/reviews/anthropic-api-adoption.md`; this file covers only the security angle. No live API call and no exploitation attempt were made. The two stdlib behaviours in F2 and F5 were reproduced offline against a closed local port with a fake key.

CPG: not applicable — plan review; the code it introduces does not exist yet, and the existing seams (`modelconfig.py`, `transport.py`, `executor.py`, `.claude/` settings) were read directly.

**Verdict: approve with suggestions.** There is no blocker: the design direction (key in a file, never in env, per-call read, fail-closed, no key in children, native wire) is sound. There are 5 majors, 8 minors and 1 nit. F1 to F5 should be folded into the plan before the unit they gate merges (F2, F5 before U1; F3, F4 before U2/U4; F1 before the first live key exists, i.e. U1's spike).

Counts: blocker 0 · major 5 · minor 8 · nit 1.

## 2. Findings

### Major

**F1. The "best effort" mitigation for agent-readable key is weaker than the plan assumes (plan §3.3 residual, R4). Confirmed from repo config.**
Evidence:
- `.claude/settings.json` allows bare `Bash` and `Edit(**)`. A `permissions.deny` entry for the key path lives in that same tracked file, which any session can edit under `Edit(**)`. The deny list today only blocks commit and PR attribution strings.
- `.claude/settings.local.json` (untracked) additionally allows `Read(/$HOME/**)` and `Bash(curl *)`. A deny on the `Read` tool path does win over that allow, but it only governs the `Read` tool. The plan's own phrase "deny entries for the key path" cannot cover Bash: `Bash(...)` deny rules are literal pattern matches on the command text, so `cat ~/.config/graph*/anth*`, `python3 -c 'print(open(os.path.expanduser(...)).read())'`, `find ~ -name '*.key' -exec cat {} +`, or a variable-built path all miss `Bash(*anthropic.key*)`. With `Bash(curl *)` already allowed, a read key can then be sent anywhere without a prompt.
- Project settings only apply to sessions whose cwd is in the repo. The decision-models `claude` children run with `exchange/` of the campaign `--root` as cwd (`campaign.py:1350`, `runner.py:child_argv`), which can be anywhere; they also get `HOME` through `CHILD_ENV_ALLOWLIST` (`runner.py:72`) and have `Read, Write` (`agents/dm-generator.md`, `dm-labeler.md`). So a repo-level deny does not reach the very children §3.3 says "never get the key". Their protection today is only that they don't know the path and have no env var.
- The root `claude/scripts/guard-*.sh` hooks are destructive-op pattern matchers with no credential-path patterns (grep for `.key`/credential/secret finds only the security-expert report-write guard).
- The user's `defaultMode: auto` classifier may refuse credential reads, but that is outside the repo's control and not a mitigation this plan can cite.
So the residual risk stands as the plan states it, but the stated mitigation is closer to "advice" than "control", and the repo is not equipped to enforce it.
Suggestions (in order of strength):
1. Hard control: the key readable only by something agents are not. Options: a separate low-privilege OS user owning the file with a tiny loopback broker that adds `x-api-key` (the holders then never read the key, and the endpoint rule lives in the broker), or at minimum a Claude Code sandbox `filesystem.denyRead` on the key directory for Bash (OS-enforced; verify the current setting name and scope in the docs before adopting).
2. Put the `Read` and `Bash` deny entries in `~/.claude/settings.json` (user scope, applies to every cwd including children) rather than, or in addition to, the editable project file; state plainly in §3.3 that a Bash deny is a tripwire, not a barrier.
3. Add one human-run live check (to U1's done list): in a session with the deny installed, prove the `Read` tool and a plain `cat <path>` are refused, and record that `python -c`/glob forms are not.
4. Do not publish the key path in always-loaded files. §3.3 and the planned `AGENTS.md`/README edits name `~/.config/graphmind-ai-lab/anthropic.key`; use `$ANTHROPIC_API_KEY_FILE` in docs and let the operator pick the path.
5. The "opt-in rule" only prevents accidents: nothing stops an agent from setting `ANTHROPIC_API_KEY_FILE` itself and running `model-bench run --hosted` (the token bound limits spend, not access). State that, and keep `--token-bound` mandatory (already planned).

**F2. Key text can reach exception messages through stdlib errors, and the planned scrub does not catch it. Reproduced.**
Evidence (run offline, fake key, closed port): `urllib.request.urlopen` with header `x-api-key: sk-ant-FAKE\nINJECT: 1` raises `ValueError: Invalid header value b'sk-ant-FAKE\nINJECT: 1'`; a key containing a non-latin-1 character raises `UnicodeEncodeError` naming the characters. The key text is in `str(exc)`, in a `repr` form that escapes control characters, so the plan's exact-value `scrub()` does not match it (the `sk-ant-[A-Za-z0-9_-]+` regex removes only the leading run). The existing `transport.py` ladder does the same thing in its rung 4 (`malformed request: {exc}`) and rung 3 (`{type(exc).__name__}: {exc}`). A key file with a trailing `\r` or stray byte (WSL/Windows editor) triggers it.
Why it matters: that message goes to `ProviderCallError`, step records, logs, ledger events, the exact AC-7 locations.
Suggestions: (a) validate the key at load against a strict pattern (`^sk-ant-[A-Za-z0-9_-]{20,}$` after stripping one trailing newline; refuse anything else with a closed message that never includes content); (b) in `post`, build every message only from the closed fields (`kind`, `status`, `error.type`), never `str(exc)`/`repr(exc)` of a `ValueError`, `UnicodeError`, `OSError` or `HTTPException`; (c) raise `from None`, or scrub the cause, so the chained cause is not printed in a traceback; (d) add a U1 test with a key containing `\n`, `\r`, and a non-latin-1 character that asserts the canary is in no raised text, no `__cause__` text and no `__context__` text.

**F3. Hosted access is not confined to switched workspaces: the `requested` rung bypasses the switch. Confirmed in code.**
Evidence: `resolve()` falls through override → `requested` → default (`modelconfig.py:~735-760`). `requested` comes from workflow node `config.model` (`executor.py:929-931`), the run's `stepModel` (`tools.py:1199`) and the guard `model` argument (`app.py:796`). A workflow definition is workspace data. Plan §3.4 item 4 refuses hosted chains only for `embedding`/`extraction` and private workspaces, so once the holder server has `ANTHROPIC_API_KEY_FILE`, any unflagged workspace whose workflow def names `anthropic/...` or a `claude-*` role sends its prompts and retrieved graph data to Anthropic with no switch at all, violating FR-2 ("an observable switch") and FR-1's spirit. The server's dev REST/MCP surface is unauthenticated by design (`falkor-chat/docs/SERVER.md` §1.3), so "who can author a workflow def" is any local caller. Same credit-burn path.
Suggestion: allow a hosted element only when it was supplied by the workspace-override rung (the observable switch), and refuse hosted for `source in {"requested", "default"}` with `ModelResolutionError` (this also covers the `services.py:743-770` publish-time validation, which resolves without `ws`). Add it to test (7) in §5. If the design wants def-level opt-in later, make it a separate explicit workspace flag.

**F4. `WorkspaceConfig.private` enforcement is weaker than stated: stale within a drive and fail-open on absence (plan §3.4 item 6, AC-10).**
Evidence: the override dict is read **once per drive** and stamped on the run (`executor.py:683`, `run["modelOverrides"]`), then passed as `overrides=` to every `resolve()` in that drive (`executor.py:932`, `tools.py:1200`, `app.py:795`). If `private` is added to that dict, a workspace flagged mid-drive keeps sending until the drive ends, so "stops sending at its next call" is false for long drives. Also `_workspace_override_ref` uses `overrides.get(key)`; a missing `private` key (older dict shape, test double, graph read error handled as "no row") reads as "not private". The requirement chose "unflagged = switchable", so the default is fail-open by decision; the implementation must not extend that to errors.
Suggestions: (a) for a hosted element, read `private` fresh from the graph at call time (one extra RO query only on hosted calls), not from the drive snapshot, or reword the claim to "at the next drive" and accept it; (b) treat an exception or a missing/ill-typed `private` key in the hosted path as private (refuse), local-only chains keep today's behaviour; (c) AC-10 tests should cover: flag after the drive started, graph read failure, dict lacking the key.

**F5. The endpoint rule is the only control keeping the key from a wrong host, and the plan specifies it in one sentence; stdlib defaults would undercut it. Partly reproduced.**
Evidence: `urllib` forwards custom headers on redirect: `HTTPRedirectHandler.redirect_request` rebuilds `newheaders` from `req.headers` (only content-length/type dropped), so `x-api-key` follows a 3xx to another host (read from stdlib source; the loopback stub or a future base URL change makes this reachable). The default opener also honours `http_proxy`/`https_proxy`: an `http://127.0.0.1` stub call with `http_proxy` set and no `no_proxy` sends the key in clear to the proxy. The base URL is also logged (`modelconfig.py:544`, `_log.info("... baseURL %s")`), so a userinfo or query in it would be logged.
Suggestions, to write into §3.3: parse with `urllib.parse.urlsplit` and compare `hostname` exactly (`https` scheme, host `api.anthropic.com`, no userinfo, no query/fragment, port empty or 443; or `http|https` plus a literal loopback IP; drop the `localhost` name, which depends on resolver/hosts file); refuse `api.anthropic.com.evil.test` and `api.anthropic.com@evil.test` by test; build a dedicated opener with `ProxyHandler({})` and a redirect handler that raises on any 3xx; certificate verification left at default (add a test that no unverified context is created). Put the check in the shared `post` so all three copies carry it, before the key file is opened (as planned).

### Minor

**F6. Hosted provider spec may carry a second, unmanaged key path.** `_build_provider_spec` reads `options.apiKey` and `options.headers` (with `{env:}`/`{file:}` substitution), and `_auth_headers` sends `resolved.headers` plus a `Bearer` header. `_infer_protocol` returns `anthropic` for any provider whose `npm` contains "anthropic", so once `anthropic` joins `SUPPORTED_PROTOCOLS`, a provider declared that way in the shared `opencode.json` (OpenCode/`tank` read it; none today, grep count 0) becomes live in falkor-chat. Suggest: for `protocol == "anthropic"`, refuse at load any `apiKey`, `headers` or `{env:}`/`{file:}` use, and ignore `_auth_headers` in the hosted branch; test it.

**F7. Leak sweep (U10, AC-7) gaps.**
- Add the real-world residue the canary design does not reach: the operator's shell history and any Claude session transcript (`~/.claude/projects/**`) from the moment the key file is created. The plan has no provisioning step; document a human-terminal `umask 077; read -rs` flow, never `echo sk-ant-… > file` in a session or typed into history.
- Sweep pytest temp dirs (`/tmp/pytest-of-*`), `__pycache__`, FalkorDB `dump.rdb`/container logs, and the holder processes' own logs and ledgers: AC-7's "excludes the closed list of holders" must exclude the key file and the holder environment only, not the holders' outputs.
- Plant-and-find must happen in a throwaway clone (planting a canary into git history in the real repo would create the leak).
- The sweep cannot see a key form other than `sk-ant-…`; add one sweep for the F2 variants (CR/LF, non-latin-1).
- Run with a pytest `--showlocals`-style failure forced once, to confirm locals are not dumped with the canary (the key sits in a local during `post`).

**F8. Key-file loader hardening (§3.3).** Open with `O_NOFOLLOW`, `fstat` the descriptor (regular file, `st_uid == geteuid()`, mode `& 0o077 == 0`), and check the parent directory is not group/other-writable, so there is no path-then-open race and no symlink swap. A key under `/mnt/c` (drvfs reports 0777) will always be refused; say so in the operator note so nobody relaxes the check. "Inside the git work tree" needs a defined test (walk parents for `.git`, resolved path) since the module takes no subprocess.

**F9. Byte-copied wire module: drift guard can silently skip.** A security-critical module (scrub, endpoint rule, key loader) in three copies guarded by a byte-equality test "that skips when the source is absent" means a standalone checkout of `decision-models/` or `falkor-chat/` can diverge undetected. Suggest a pinned SHA-256 constant per copy checked by the copy's own tests (no cross-component read), plus the existing equality test as a second line. Also keep the F2/F5 tests identically duplicated in each component so a divergent copy fails locally.

**F10. Provider-controlled text in "closed" messages.** The fatal sentence names `error.type` (§3.4 item 3, §3.1). `error.type` is read from the response body; a stub, a proxy or a future provider change can put arbitrary text there. Map it through an allowlist of the documented enum values (`authentication_error`, `permission_error`, `rate_limit_error`, `overloaded_error`, `billing_error`, `invalid_request_error`, `not_found_error`, `api_error`), else `unknown`. Keep `ProviderCallError.detail` (which the existing transport fills with the response body, `transport.py:~150`) `None` for hosted errors; add a test.

**F11. Public-facing credit burn.** FR-14 leaves interactive use unbounded by decision. The storefront is reachable by any participant, so a switched `demo` workspace lets an anonymous caller spend credits with long conversations. Not a plan defect, but state it as accepted residual risk in §3.4 and add cheap guards that do not contradict FR-14: a conservative default `max_tokens` per hosted call, and the existing storefront turn limits documented as the ceiling. Customer free text also goes to Anthropic in that mode: demo data only, which is the FR-15 premise; the operator should know it when exposing `/shop` publicly.

**F12. Hosted bench packs "synthetic by construction" is asserted, not checked.** I verified the embedder corpus is a seeded synthetic script (`seed_eval_corpus.py` provenance) and the join-in pack declares `inputs: synthetic`, but the other role packs (`guard-judge-understanding`, `nlq-structured-query`) are copied from falkor-chat test goldens with no such declaration. Suggest `run --hosted` refuse a pack whose `pack.json` lacks an explicit `inputs: synthetic` (or equivalent) marker, so a future real-data pack cannot be sent by accident.

**F13. Campaign `--root` is not ignored.** `decision-models/.gitignore` ignores only `.venv/`, and the root `.gitignore` has no ledger/store rule, so a campaign root inside the repo would have `ledger.jsonl`, `items.jsonl`, `req/`, `ans/` and `runs/*.err` tracked (the requirements' consult flagged this). The planned `api-call` event is safe (batch, model, token counts, hosted flag: no key, no prompt, no header, no request id), but any `apirun.py` failure text written to `store/refusal` or a ledger reason must come only from the closed wire messages (F2). Add a `store/` and `exchange/` ignore, or refuse an in-repo `--root` for `api-run`.

### Nit

**F14.** The plan's §3.3 sample path and the `*.key` `.gitignore` rule: also add `.env.*` and `*.key.*` patterns for editor backups (`anthropic.key~`, `.swp`) if the key file ever sits near the repo; the in-repo loader refusal already makes this a backstop only.

## 3. What's solid

- Key held in a file, not env (no `/proc/<pid>/environ`, no inheritance), read per call, no object retention, rotation/revocation without tracked-file edits.
- Opt-in per process through a path variable, fail-closed when unset (FR-1/AC-1); key never in the child allowlist; `Secret` repr stays masked.
- Fatal vs fallbackable split (`auth`/`request` fatal) closes the "revoked key silently answered by local model" hole (AC-12); `request` as fatal is the right call.
- Decision-models `api-run`: no `claude` child, no tools in the call, so no Read/Write exposure for the key during API generation; ledger event set is minimal and carries no prompt, header or id; pin mismatch refuses rather than re-pins.
- Bench hosted fingerprint profile with forbidden LM Studio fields and no org/workspace/request-id field; transcripts directory already ignored; run records hold only the named fields.
- The `fatal` flag is additive and local providers never set it; embedding/extraction refusal at `resolve()` is the right choke point.

## 4. Open questions

1. (`devops`) Is a separate-user broker, or the Claude Code sandbox `denyRead`, acceptable for the key directory (F1)? This decides whether R4 is closed or merely documented.
2. (stakeholder via `teco`) F3 narrows hosted use to the workspace-override rung; confirm no use case needs a workflow def to name a hosted model directly.
3. (`architect`) F4: fresh per-call read of `private` for hosted calls, or accept per-drive staleness and reword?
4. (`qa-engineer`) Fold F2, F5, F7 variants into the U1/U10 test list.

## 5. Exploitation attempts (FR-10)

None. Two offline stdlib reproductions only (header-value `ValueError`/`UnicodeEncodeError` text; redirect handler source) against a closed loopback port with a fake key; no live target, no API call.
