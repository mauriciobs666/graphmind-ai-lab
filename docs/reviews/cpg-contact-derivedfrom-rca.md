# RCA: cpg contact.json `derivedFrom` is not an ancestor of HEAD

> **Status:** active · **Owner:** `analyst` · **Tracks:** —

Scope: `cpg/tests/test_contact.py::test_contact_json_matches_rederivation_at_its_derivedfrom_tree`, `cpg/spikes/frontend-fitness/make_contact.py`, `cpg/spikes/frontend-fitness/contact.json`. Read-only; re-derivations ran in-memory, no file written, no git state touched.

## 1. Symptom and impact

Run from `cpg/` with `.venv/bin/python -m pytest tests/test_contact.py::test_contact_json_matches_rederivation_at_its_derivedfrom_tree`: 1 failed.

```
E   AssertionError: derivedFrom ebe66607cbbc432249305d5fb9e9c236cbef8293 is not an ancestor of HEAD
```

The drift guard for the S59 contact list is red on `main`. It is not a content problem (see §3). It has been red since the history rewrite of 2026-11-04 22:09-22:25 -0300, well before the 2026-11-09 path-cleanup commits.

## 2. Reproduction and evidence

- Reproduced as above (exit: 1 failed in 0.14s).
- `make_contact.check()` (`make_contact.py:334-348`) reads `derivedFrom` from the file, requires a 40-hex sha, requires `resolve_sha(sha)==sha` (object exists), then `git merge-base --is-ancestor sha HEAD`, then re-renders from that tree and compares to the file text. The failing step is the ancestor check (line 344-345).
- `derivedFrom` = "HEAD at generation" (`make_contact.py:16`, set at line 265 via `resolve_sha(treeish)`); `--check` re-derives from exactly that tree so the file is verified against the tree it was built from, not a moving HEAD. The ancestor requirement exists to ensure the pinned tree is part of this repo's lineage.
- `git cat-file -t ebe66607...` -> `commit` (object still present). `git merge-base --is-ancestor ebe66607 HEAD` -> exit 1. `git branch -a --contains ebe66607` -> empty. It is reachable only from the reflog (`HEAD@{405}`, subject "docs/plans: ledger", 2026-11-04 19:04:58). Repo is not shallow (`--is-shallow-repository` false).
- `git log -- cpg/spikes/frontend-fitness/contact.json`: `16fa80b9` (2026-11-04 19:13:14, "S59 contact list per ML v1.9") is the commit whose blob carries `derivedFrom: ebe66607...`. It is an ancestor of HEAD. Its pre-rewrite twin is `67cbcc03` (`HEAD@{404}`).

## 3. Causal chain

1. The S59 builder committed contact.json on 2026-11-04 19:13 recording `derivedFrom = ebe66607` (the then-HEAD, a ledger commit).
2. At 2026-11-04 22:09:14 a `rebase (start): checkout refs/remotes/origin/main` replayed ~357 local commits onto upstream (`HEAD@{357}`..`HEAD@{189}`, finish 22:25:11). Every replayed commit got a new hash: `ba24b19a`->`a8a4c7f2` (ML note v1.9), `ebe66607`->`edb6adf1` (ledger, same timestamp 19:04:58 and subject), `67cbcc03`->`16fa80b9`.
3. A rebase rewrites commit hashes but not file contents; contact.json still names the old sha `ebe66607`, which now exists only as an orphan (reflog-only, gc-eligible).
4. `check()` fails at the ancestor guard. The test fails deterministically.

Rewritten twin: `edb6adf1` is an ancestor of HEAD and is the direct parent of `16fa80b9` (as `ebe66607` was of `67cbcc03`).

Class check (observed, via in-memory `derive`/`render` from `make_contact.py`):
- `render(derive(treeish=ebe66607)) == contact.json` text: **True**. The list is not wrong against its original tree.
- `render(derive(treeish=edb6adf1))` differs from the file in exactly one line (261-ish, line 259 of diff): the `derivedFrom` value. So the contents (file lists) derived from the rebased twin are identical; only the pin differs.
- Trees differ (`ebe66607^{tree}` a5c3c385, `edb6adf1^{tree}` db9aea14) because the rebase picked up upstream content (a 9-file diff incl. `docs/requirements/work-item-numbering.md`), none of it in the derivation's source set.

Ruled out:
- Environment (shallow/missing objects): repo is full, object present, so a "not a commit here" failure would have been the other message.
- Genuine regression: derivation from the original tree matches byte for byte; from the twin matches except the pin.
- The 2026-11-09 path cleanup (`eca54ba2`..`b9647bf9`): the pin was already orphaned on 2026-11-04. I did not run the test at an earlier commit; the conclusion rests on the reflog chronology and the ancestry result, which do not depend on later commits.
- The earlier `filter-branch: rewrite` (2026-10-31, `HEAD@{673}`) predates the S59 commit, so it cannot have orphaned a sha recorded on 11-04.

## 4. Root cause

**Stale pinned sha: history rewritten (rebase onto origin/main) after the sha was recorded.** Confidence: *confirmed* (reflog shows the rebase and the old/new pairs; the twin's re-derivation reproduces the file apart from the pin).

- Trigger: the 2026-11-04 rebase of local commits onto `origin/main`.
- Underlying flaw: the pin identifies a tree by commit hash, which is not stable under rebase/amend/filter-branch; the guard enforces "is an ancestor of HEAD", so any rewrite of the pinned commit's lineage turns it red even though content is intact.
- Contributing: the repo's workflow rewrites history (filter-branch on 10-31, rebase 11-04); the file is self-referential (the commit that carries the pin is written after the commit pinned); no one ran the cpg suite after the rebase.

## 5. Suggested fix and prevention (owner: `tdd-engineer`)

Options:
- **A (minimal, recommended now):** regenerate contact.json with `derivedFrom = edb6adf10935a99d67a8cf1d2add7cb3f4da4dd3` (one-line change, content otherwise identical, verified in-memory above) via `make_contact.py --from edb6adf1`, then run the test. Cheap, but it will rot again at the next history rewrite.
- **B (durable):** pin a tree hash (`git rev-parse <commit>^{tree}`) or a hash of the derivation's source blobs instead of a commit sha, and have `check()` verify that tree object exists / re-derive from it, dropping the ancestry check. Rebases that do not touch the source set keep the pin valid. Cost: the tree sha still changes if upstream content in the derivation's source set changes, and the orphaned-commit object can be gc'd; with a content-hash of the source blobs this disappears. This requires a `VERSION` bump in the JSON schema and updating `test_contact.py:63,80,282-331`, which currently assert a sha-shaped `derivedFrom` and the "not an ancestor" negative case.
- **C:** keep the ancestor rule but have the test skip with a clear message when the sha is unreachable. Not recommended: it turns a real drift guard into a silent pass.

Recommendation: A now (unblocks the suite), B as a follow-up if history rewrites are expected to continue. Design decision belongs to the owner. Prevention: run `pytest` in `cpg/` after any rebase/filter-branch, and a post-rewrite check that `git merge-base --is-ancestor $(jq -r .derivedFrom ...) HEAD` holds.

Not verified: I did not run the full cpg suite, only the named test, and did not test the fix by regenerating the file (done in-memory only).
