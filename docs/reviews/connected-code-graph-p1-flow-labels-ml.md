# ccg P1-34 python flow labels: transcript check ruling (prereg §2.3)

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** C-1001 (P1-34 U5) · **Extends:** `docs/plans/connected-code-graph-p1-flow-prereg.md`

## Verdict: labels stand with recorded deviation

No redo. The literal hits are not a §5 row 1 "transcript-check hit" in intent, and the labels are sound on every mechanical check.

## 1. The literal hits (question 1)

I parsed the `tool_use` inputs of the nine Bash calls (the tenth is the hand-back). The labeller read only the sample (calls 1-3, 6), wrote builder scripts to the session scratchpad (4, 5, 7), listed directories and installed package names (8), and ran `flowcheck` (9). Read against the intent of §2.3:

- **Scratchpad builders (`h.py`, `b1-b3.py`).** They build and append the JSON lines to the output file. They read nothing and analyse nothing. The allowlist exists to catch reading code, an analysis tool or outcome knowledge; writing the output through a helper is none of them. The labels are the labeller's own judgement either way.
- **`ls -d .venv*` and `pip list | grep -i tree`.** They return directory names and package names. They expose no corpus, held-out, outcome or projection data. The motive (a `flowcheck` failure) is a repair of the instructed command, and the venv python is the only interpreter on which that command runs.
- **Leakage.** None of the 18 forbidden substrings occurs, no corpus file was opened, no tool other than Bash was used, and nothing outcome-bearing was seen. The void rule protects the blindness of the labels; that is intact.

The deviations are real and are recorded, not silenced: (a) three scratchpad writes and two environment-listing commands are outside the exact-path allowlist; (b) `flowcheck` ran as `cpg/.venv/bin/python …`, not `python3 …`.

## 2. Labelling from `source` alone (question 2)

A protocol deviation, not a validity problem. §2.2 says the labeller "may read" the corpus; it does not require it. The sample `source` equals the corpus span for all 30 methods: lines `lineStart..lineEnd` of the file, compared mechanically, are identical except that the first line is stripped of its leading indentation (indented methods; 16 of 30). Labelling from `source` therefore loses no information inside the span. The brief's "file wins" clause was never exercised, which is vacuous here. What the labeller could not see is context outside the span (enclosing class, module imports). That limits only how it names a callee it cannot resolve from the span, and it is the same view the span-level projection is scored on. State it as a caveat on `unsure` pairs, not as a defect.

## 3. Mechanical audit (question 3)

Counts only. 30 sample lines, 30 label lines, ordinals and `qname` aligned for all 30. 334 pairs (228 data, 106 control).

| Check | Violations |
|---|---|
| endpoint `line` outside `lineStart..lineEnd` | 0 |
| `cond` code not a literal substring of the source | 0 |
| pair with source equal to sink | 0 |
| call ordinal not below the textual count of `callee(` in the method | 5 |

The 5 are one endpoint, `calleeName` `<dynamic>`, ordinal 0, at line 2089 in method 16 (`getattr(...)(*args, **kwargs)`, a call whose callee is not a name). A textual count cannot see it, so it is consistent and not a violation. Effective violations: 0. `flowcheck` independently reports 0 errors, 14 unsure, complete (labeller's report; not re-run here).

## 4. Recorded deviation for the ledger

> U5 labeller transcript: zero hits on the 18 forbidden substrings; no corpus file opened. Literal exact-path deviations, ruled not voiding by `data-scientist` (`docs/reviews/connected-code-graph-p1-flow-labels-ml.md`): scratchpad builder scripts that only write the output file; `ls -d .venv*` and `pip list | grep -i tree`; `flowcheck` run with `cpg/.venv/bin/python` because `python3` lacks `tree_sitter`. Labels stand.

## 5. Amendment for any further dispatch (second file, re-dispatch)

Record as a ledger amendment to §2.2/§2.3, without editing the committed prereg: (1) the `flowcheck` command is `cpg/.venv/bin/python cpg/spikes/frontend-fitness/run_fitness.py flowcheck python --set p1` (the system `python3` has no `tree_sitter`); (2) the allowlist adds the session scratchpad directory for helper scripts that only write the named output file, and `ls`/`pip list` on environment directories are not hits; (3) the substring list is unchanged and still decides.

## 6. Risk

The venv-python and builder-script allowance should be applied identically to every later dispatch check, or the rule becomes selective. Any hit beyond these two classes stays a void.
