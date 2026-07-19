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
