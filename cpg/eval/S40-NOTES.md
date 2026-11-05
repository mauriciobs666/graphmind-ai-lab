# S40 — evaluation kit, briefs and manifests: result notes

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** P0-32 (S40)

Ground-truth custodian run, 2026-11-05. No key was written. Plan: `docs/plans/connected-code-graph-p0-2.md` S40 row and §6.2; protocol: `docs/plans/connected-code-graph-eval-protocol-ml.md` v1.4 §2.4, §3.3, §3.6, §13.1, Appendix B. Every figure below was read from command output of this run.

## Pins and hashes

| Item | Value |
|---|---|
| `EVAL_COMMIT` (`cpg/eval/EVAL_COMMIT`) | `cf7b2042cce7bad5d0ab7876940bcea00343d914`, the HEAD at the start of the run (S48 audit hardening); an ancestor of HEAD; `git status` showed no change under `cpg/eval` at that point. The coordinator commits the new files afterwards. |
| Eval dir | `~/.local/share/ccg-eval/cf7b2042cce7/` |
| Kit | `kit-src/`, 1448 files; manifest `files` array (sorted path + sha256) hashes to `a87dcd2112a6f36ee8e2af424761b385173bb7c5077cbac6aafd76aef56a7390` (sha256 of its canonical JSON); re-hash of every file on disk: 0 mismatches, 1448 on disk |
| `cpg/eval/kit-manifest.json` | sha256 `6d9132fd2c4196054aac67939d24391196048d63c1573586c4513ae16479614d` (after `--update-briefs`; carries 13 brief hashes) |
| Flow anchors | `work/flow-anchors.jsonl`, sha256 `a2e9cac3d73def5c13b617e64ee129a01f0c21bf0e539bc8e77d3f04a3288f40` |
| Grammar extract | 12145 bytes, sha256 `5202c983c7ced1bc2b1ef166fd2b8a77bfbf3d444f1880e3a5afdd06e009f3ff`, equal to S52's pin, in all 13 briefs |
| `bundlePySha256` | `18be67564e0d20f4d8f2478fd53ca77680fc9abe1ddd537179dc6be0cf1d427e` (`git show EVAL_COMMIT:cpg/ccg/bundle.py`; equal to the working-tree file) |

## What was run

1. `export_kit.py --repo . --eval-commit <E> --manifest cpg/eval/kit-manifest.json`: rc 0; files 1448, path-excluded 863, content-check-excluded 16 (class doc 13, prompt-md 3), excluded-scope 3. Independent checks: `grep -rilE` of the content pattern over `kit-src` gives 0 files; no `.git`, no symlink, no writable path.
2. Seed (EP §3.6): `ccg-p0f-flow-v1|python|<E>` and `ccg-p0f-flow-v1|javascript|<E>` (full 40-hex `<E>`). `--draw-flow-anchors`: 16 anchors, 8 python and 8 javascript, skipped-files 0, error-trees 0; a second run produced a byte-identical file. The list is recorded in `eval-manifest.yaml` `flowAnchors` (hash, seeds) and pasted verbatim (compact JSON array) into every key-writer brief.
3. `--list-units`: 6 units. `--list-cypher-tools`: `query`.
4. Briefs: 13, generated from Appendix B (B.0 plus B.1 / B.2 / B.3) and the extract taken from S52's `agents.json`, then re-verified from the files in the role roots by an independent script: B.0 and the role part verbatim with slots filled, no foreign part, no unfilled slot, extract length and hash equal in all 13, the flow list in the brief equals the file, the only `§` signs are the three inside the verbatim text. Sizes 18,333 to 24,495 bytes. Roles and strata: KA and KB x {`http`,`mcp`,`scripts`,`wiring`,`flow`} (10), plus QW, AN1, AN2 under stratum token `impact` (the launcher requires a stratum; the plan names none for them, so this is my naming choice and reversible).
5. Role roots: `run-writer.sh prepare <E> <role> <stratum> --brief ...` for all 13 (writes only under the eval dir; kit copies diff-identical to the master). Done because `export_kit.py --update-briefs` hashes `roles/*/BRIEF.md` and `prepare` refuses a pre-existing role root, so the two cannot be reached any other way. To change a brief later, move that role root aside and re-run `prepare` and `--update-briefs`.
6. `cd cpg && .venv/bin/python -m pytest -q tests/test_eval_tools.py`: 230 passed before the run and 230 passed again after the last file was written.

## S52-NOTES cuts mirrored

The extract is taken byte-for-byte from the key-writer definition (`agents.json`), so rule 7 is gone, the "(Rules 5 and 7 are not used in this brief.)" line, `http` step (7) without the catch-all clause and the name/section cuts are exactly S52's. The brief-only authored text mentions no other role, no plan or protocol document and no learning/KB instruction. The `ccg-annotator` definition leaves `{outputPath}` to the brief: the AN briefs set `out/AN1.jsonl` / `out/AN2.jsonl`.

## Slot fills (the one place a paster chooses a form)

`{units}` and `{flowAnchors}` are single-line compact JSON arrays inside the existing backticks; `{cypherMcpTools}` is `query`. The flow list is also filled in non-flow key-writer briefs (the B.1 `flow` bullet carries the slot in every brief).

## Open items for the method owner

1. **`screen.specVersion`.** EP2 §1.5 says "spec3's version"; spec3 (`Version: 1.2`) is superseded by spec4, which has no `Version:` field. I recorded `"1.2"` with the document named beside it. Ruling wanted: keep spec3's 1.2, or name spec4.
2. **`falkor-chat/server` ports `[8000, 8200]`.** 8000: `falkor-chat/compose.yaml` service `server` (container side), `falkor-chat/Dockerfile` `EXPOSE 8000` and `--port 8000`. 8200: only the default of `AGENT_TEAM_PORT` in `falkor-chat/scripts/start_agent_team.sh` (lines 137, 157), which runs the same `falkorchat.app`; it is not a literal `--port N`, so it is a reading of EP §2.4's start-script clause. It is what lets `.mcp.json`'s `falkor-chat-agent-team` (`:8200`) and the Kiro agent's `:8000/mcp` both map to a listed unit. Drop 8200 if the strict reading is wanted. The other five units have empty `ports` (no `EXPOSE`, `--port` or `port=` listener found by grep in their source; `cypher-mcp/Dockerfile` states it listens on nothing).
3. **Grain-table re-check at `EVAL_COMMIT` (EP §2.4):** `.mcp.json`, the Kiro agent file (`toolsSettings` `{}`, `includeMcpJson` `false`), `.claude/settings.json` (no `hooks` key), `mcp-monitor/config.example.toml`, `falkor-chat/compose.yaml` and `falkor-chat/server/.env.example` still match their rows. Every other JSON/YAML/TOML file in the kit falls to the "any other" row; no row was added. Only `FALKORDB_PORT=6379` is a port-like `.env.example` value, and 6379 is in no listed unit's `ports`.
4. **Briefs for QW/AN** carry no per-cell count table beyond B.2's own counts, and the AN briefs omit the dependency rubric (they judge ambiguity, not dependencies). `QW-nogold.jsonl` is placed later (S43a).
5. Qualifier: the working tree has uncommitted changes elsewhere (`decision-models/`, `cpg/spikes/`); the kit is `git archive` of the commit, so none of them is in it.
