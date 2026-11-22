# P1-34 flow re-measurement: U4 first-label barrier (methodology review)

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** C-1001 (P1-34 U4) · **Reviews:** `docs/plans/connected-code-graph-p1-flow-prereg.md` §3–§4, `docs/plans/connected-code-graph-p1-flow-extract-pin-ml.md`

## Verdict: approve with suggestions

U5 (first label) may proceed once suggestion 1 is recorded. No blocker. No held-out data was opened, no draw was run, no record was edited.

## Pass 1

**(1) Filled brief — pass.** `sample/python-flow-p1.brief.md` is byte-identical to the output of `run_fitness.fill_flow_brief('python')` with flow-set `p1` (recomputed in memory, no write), and `validate_flow_brief` returns no errors. Its diff against `sample/python-flow.brief.md` is exactly four lines, all the sample/label path slots (`{N_METHODS}` is 30 in both). The template sha (`95dcddd1…`) equals the record pin. The leak patterns of `FLOW_LEAKS` do not match, and the one `Joern` mention is the prohibition sentence; the plain words `results`/`result` that remain are the worked-example prose. The brief names no Joern projection, cell, join, decision cell, threshold, recall figure or held-out item. The sample rows carry only `ordinal, path, qname, lineStart, lineEnd, source`, so no outcome-correlated field exists.

**(2) Draw record — pass, with a verifiability limit.** Line 9 of `cpg/eval/preregistration.jsonl` (`ccg-p1-flow-python-v2-draw`; the id is `-v2` because `-v2` superseded `-v1` per the extract-pin note) holds every §1.2 field.
- Recomputed and equal: `sampleSha256` `6f4e1f98…7208` and `briefSha256` `618ea5b4…3e33`.
- The pins in `-v2` equal the tree: `s28_core.py`, `extract.sc` (`02001899…`), `s28_run.py`, `s28_metrics.py`, `run_fitness.py`, `flow_decide.py`, `flow_p1_audit.py`, brief template.
- Cannot verify: `tier2Sha256` `07170f2e…5d1d` (the file is private and outside the repo, by design), `excludedTier1Frame` 2 and `excludedTier2Frame` 0 (they depend on the private exclusion file), the tree-sitter 0.25.2 and grammar 0.25.0 versions as run on the owner's machine, and `devAnchorOverlap` 0 (`teco`'s count; I did not recompute it). These rest on the custodian's report.

**(3) Sample — pass, on what is checkable without the private file.**
- 30 lines, ordinals 1–30, 30 distinct keys; bytes equal `jdump` of the rows (the tool's canonical form).
- Overlap with `sample/python-flow.jsonl` is empty on `(path, qname)`, on `(path, span)`, and on span intersection within a file.
- The 30 rows' hashes `sha256(seed|path:lineStart:qname)`, with the seed `ccg-p1a-v1-flow|python|0caf85e3…`, are strictly ascending, so ordinals follow the hash order.
- By reading the code: `cmd_flowdraw_p1` builds the seed with `SEED_PREFIX_P1` and the corpus commit, subtracts the P0 `siteKey`s, asserts (368, 338, 163) before exclusion, filters and then `hash_sort`s with ties broken on the key, and takes the first 30. The seed rule, `siteKey`, the order and the filter-before-sort all match prereg §1.1.
- I did not recompute the population order or diff it against the sample: doing so would reveal which methods the exclusion skipped (prereg §3.4 residual). Equivalence to the actual run therefore rests on the tool being the pinned `run_fitness.py` (sha equals the record) and on the custodian's counts.

**(4) §4 U5 prerequisites — nothing missing.** Sample, brief and `-draw` record are committed and no `labels/python-flow-p1.jsonl` exists. The dispatch text (§2.2) and the `flowcheck --set p1` command are specified.

## Suggestions

1. **Custody deviation, record it (minor).** The `-draw` record's `custodianNote` says a coordinator agent ran the draw as the owner's delegate, so an agent session read the held-out and reserve anchor list. Prereg §3 specifies the repo owner. The measurement is unaffected (the exclusion only protects the held-out set from the labelled sample, and the labeller never sees the private file), but the same coordination context will later dispatch labellers and build the adjudication list. Say so in the decision record, and keep the private file out of every U5/U8 allowlist (it is outside the repo; the §2.3 allowlist already excludes it).
2. **Prereg text still cites `-v1` (nit).** Prereg §0 finding 3 and §7 cite `-v1` pins; the extract-pin note §3 item 4 already schedules the swap to `-v2`. Do it at the next ordinary revision, not before U5.
3. **Tier-2 release check (nit).** At release `teco` must compare the file's sha256 with `07170f2e…5d1d` (prereg §5 row 1); it fires only if the labelled pair count is below 120.
