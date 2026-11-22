# Connected code graph P1: S36 spot-check redo (custodian counts)

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** P1-37 (P1)

Counts and item numbers only; no item content. Input to the P1-37 gate report (`docs/plans/connected-code-graph-p1-gates.md` P1-37; `docs/plans/connected-code-graph-p1-ml.md` §9 "S36 spot-check"). Protocol: `docs/plans/connected-code-graph-eval-protocol-ml.md` §7.3, §7.4.

## Sample and marks

The sample is the P0 draw of 40 union items (seed `ccg-p0f-spot-v1|<EVAL_COMMIT>`, `spot.jsonl` hash prefix `379629ac97ae4cc4`): 8 per stratum, 20 singleton and 20 doubleton.

| Stratum | Items | Singleton | Doubleton | Marked incorrect |
|---|---|---|---|---|
| http | 8 | 4 | 4 | 0 |
| mcp | 8 | 4 | 4 | 0 |
| scripts | 8 | 4 | 4 | 1 (singleton) |
| wiring | 8 | 4 | 4 | 0 |
| flow | 8 | 4 | 4 | 0 |
| pooled | 40 | 20 | 20 | 1 |

Marks: 39 correct, 1 incorrect, 0 unsure. The incorrect item is number 26, with a `key-error:` note and no corrected locator. Three correct items (1, 9, 37) carry `doubt:` notes (3 doubt notes); per the coordinator's ruling they stay as marked, disclosed, and are not flipped. Wilson 95% (as in the P0 report): scripts 1 of 8 [0.022, 0.471]; other strata 0 of 8 [0.000, 0.324]; pooled 1 of 40 [0.004, 0.129].

## Escalation (EP §7.4, as written)

- A stratum with 2 or more of its 8 incorrect: no (maximum is 1, in scripts). Not triggered.
- Pooled 4 or more of 40: no (1). Not triggered. No singleton review.

## Split of the sample

| Split | Items |
|---|---|
| dev | 8 |
| held-out | 25 |
| reserve | 6 |
| removed at P0 | 1 |

By stratum, class and split, item 26 is scripts, singleton, removed at P0 (it sits in no dev, held-out or reserve list). It is the same item the P0 AI analyst marked incorrect and removed (`work/corrections.jsonl`, 1 row). The stakeholder-side redo confirms that call.

## Removal and floor

- No dev, held-out or reserve list contains item 26, so no list changed; the match file (`work/s45/match-corrected.json`) already lacks it. `apply-corrections` was not re-run: it would only repeat the P0 removal under the verifier label "analyst (AI)" and change a manifest-hashed file for no effect.
- Floors: no stratum count changed, so the floor of 8 is not affected (wiring remains at exactly 8). No reserve cluster moved; the shortfall rule did not apply.
- No held-out or reserve list changed (plan §9). The manifest was re-recorded on 2026-11-22 as a hash refresh of one raw artifact only: `runs/index.jsonl` (the headless-launcher run log, appended to by the dev2 writer sessions after S47), over the same 70 raw artifacts; the five held-out lists and the reserve list matched before and after, `split_manifest.py verify` now exits 0 (checked 77, mismatches 0), and the manifest sha256 went `ce2bb8b1…` to `d122963e…`.

## Time and disclosures

- The reviewer took about 4 minutes; the packet estimated 3 to 4 hours.
- The redo and the key-error confirmation were AI-only: the stakeholder delegated the redo to an AI reviewer and accepted its key-error call. Together with the P0 AI-analyst check, no item of this 40-item sample was checked by a human. The P0 stakeholder check of the 8 Portuguese gold items is separate and not part of this redo.
- Reserve counting items, by the shortfall rule: none used.
