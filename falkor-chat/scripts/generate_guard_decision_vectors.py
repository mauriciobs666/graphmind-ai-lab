#!/usr/bin/env python3
"""Generate the guard decision point's golden vectors (typed-decision-models v1.3 §3.2.2).

Human-invoked maintenance script, never run by the server or the test suite. It writes
`render-guard.jsonl`, `handoff-parse-guard.jsonl` and `normalize-turns.jsonl` into OUT_DIR, with
every `expected` value produced by falkor-chat's REAL code — never by the model-bench port the
vectors pin:

  * `render-guard.jsonl` — `app._render_judge_user(condition, understanding, recentTurns)`;
  * `handoff-parse-guard.jsonl` — a judge reply through the production judge
    (`app._build_llm_judge` over a stub LLM returning `reply`) then `guards._coerce_verdict`
    (`advance`); `parseFailure` is `llm.extract_own_line_json_object(reply,
    require_key="decision") is None`;
  * `normalize-turns.jsonl` — `guards._recent_turns(thread)`, the turn normalization the guard
    judge and (v1.3 §3.2.1) the join-in renderer share (typed-decision-models impl review P2-F3).
    Its threads are lists of row objects only — the item domain model-bench and decision-models
    render; `_recent_turns`' tolerance of a non-list thread or a non-object row is not pinned.

The cue/negator probe rows are emitted by iterating `guards._NEGATION_CUES` and
`guards._NEGATORS` themselves, so a change to either table changes the rows.

model-bench owns the canonical copies. After running, copy the files to
`model-bench/tests/fixtures/decision-contract/`, then byte-copy them to
`falkor-chat/server/tests/data/decision-contract/` (`server/tests/test_decision_contract.py`
asserts the two equal).

Run with the server venv, from `falkor-chat/server`:

    .venv/bin/python ../scripts/generate_guard_decision_vectors.py OUT_DIR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from falkorchat import app, guards, llm

# --- render-guard.jsonl ---------------------------------------------------------------------------

_CAP = app.JUDGE_USER_MAX_CHARS
_CAP_CONDITION = "enough info?"
#: len("CONDITION: enough info?") + len("\n\n") + len("RECENT TURNS (context only):\n").
_CAP_FIXED = len(f"CONDITION: {_CAP_CONDITION}") + 2 + len("RECENT TURNS (context only):\n")


def _turn(speaker: str, text: str) -> dict[str, str]:
    return {"speaker": speaker, "role": "user", "text": text}


def _one_turn_totalling(total: int) -> list[dict[str, str]]:
    """One turn `u: <text>` making the uncapped message exactly `total` chars."""
    return [_turn("u", "t" * (total - _CAP_FIXED - len("u: ")))]


def _two_turns_totalling(total: int) -> list[dict[str, str]]:
    """Two turns (`a: …`, `b: …`, one `\\n` between) making the uncapped message `total` chars."""
    text_chars = total - _CAP_FIXED - 2 * len("a: ") - 1
    older = text_chars // 2
    return [_turn("a", "o" * older), _turn("b", "n" * (text_chars - older))]


_LONG = "x" * 390

RENDER_CASES: list[tuple[str, str, dict[str, Any], list[dict[str, str]]]] = [
    ("condition only: no evidence blocks", "the user has provided enough information", {}, []),
    ("understanding: nested keys sorted, indent 2, list order kept",
     "the user has provided enough information",
     {"request": "Find the outage cause", "missing": ["time window"],
      "known": ["service: checkout", "region: eu-west"], "meta": {"z": 1, "a": [True, None, 2.5]}},
     []),
    ("understanding: non-ASCII escaped (json.dumps default ensure_ascii)",
     "o usuário já disse o que precisa",
     {"request": "Le café est fermé — pourquoi?", "known": ["ação"]}, []),
    ("turns: speaker: text lines, oldest first, role unread",
     "the customer confirmed the order",
     {},
     [_turn("Ana", "I want two cakes"),
      {"speaker": "salesperson", "role": "assistant", "text": "Confirm two chocolate cakes?"},
      _turn("Ana", "Yes, confirmed.")]),
    ("turns: a turn with no speaker renders 'member', no text renders empty",
     "the user answered",
     {},
     [{"role": "user", "text": "hello"}, {"speaker": "Bo", "role": "user"}]),
    ("turns: multi-line and non-ASCII text kept raw",
     "o usuário já disse o que precisa",
     {},
     [_turn("Zoë", "linha um\nlinha dois — ação")]),
    ("both blocks: understanding then turns",
     "enough info?",
     {"request": "r", "missing": []},
     [_turn("u", "t")]),
    ("cap: oldest turns dropped first until the message fits 6000 chars",
     "enough info?",
     {},
     [_turn(f"s{i}", f"{i:02d}" + _LONG) for i in range(16)]),
    ("cap: understanding over 6000 chars and a turn that cannot fit -> base hard-truncated",
     "enough info?",
     {"request": "y" * 6100},
     [_turn("u", "newest")]),
    ("cap: understanding over 6000 chars and no turns -> base hard-truncated",
     "enough info?",
     {"request": "y" * 6100},
     []),
    ("cap boundary: one turn totalling exactly 6000 chars is kept",
     _CAP_CONDITION, {}, _one_turn_totalling(_CAP)),
    ("cap boundary: one turn totalling 6001 chars is dropped",
     _CAP_CONDITION, {}, _one_turn_totalling(_CAP + 1)),
    ("cap boundary: two turns totalling exactly 6000 chars are both kept",
     _CAP_CONDITION, {}, _two_turns_totalling(_CAP)),
    ("cap boundary: two turns totalling 6001 chars drop the oldest only",
     _CAP_CONDITION, {}, _two_turns_totalling(_CAP + 1)),
]

# --- handoff-parse-guard.jsonl --------------------------------------------------------------------


def _verdict(rationale: str) -> str:
    return json.dumps({"decision": True, "rationale": rationale})


#: Hand-written rows: string versus bool `decision`, contradiction, negation edges, wrapping, the
#: own-line parse's rules, and malformed input.
REPLY_CASES: list[tuple[str, str | None]] = [
    ("bool true, clean rationale: advance",
     '{"decision": true, "rationale": "The user gave the service and the time window."}'),
    ("bool false: suspend", '{"decision": false, "rationale": "The time window is missing."}'),
    ("string \"true\": not an advance",
     '{"decision": "true", "rationale": "The user gave the service."}'),
    ("string \"false\": not an advance", '{"decision": "false"}'),
    ("integer 1: not an advance", '{"decision": 1, "rationale": "ok"}'),
    ("null decision: not an advance", '{"decision": null}'),
    ("whole-reply object with no decision key: parsed, not an advance",
     '{"rationale": "no decision key"}'),
    ("bool true, no rationale: advance", '{"decision": true}'),
    ("bool true, non-string rationale read as empty: advance",
     '{"decision": true, "rationale": 42}'),
    ("contradiction: 'not enough' at the start",
     '{"decision": true, "rationale": "Not enough information was given."}'),
    ("contradiction: 'still missing'",
     '{"decision": true, "rationale": "The order id is still missing."}'),
    ("contradiction: 'cannot '",
     '{"decision": true, "rationale": "The agent cannot confirm the date."}'),
    ("contradiction: upper-case cue is lowered first",
     '{"decision": true, "rationale": "Details are UNCLEAR."}'),
    ("contradiction: cue with decision false stays a suspend",
     '{"decision": false, "rationale": "Not enough information."}'),
    ("negation: 'no ' before 'more info' affirms",
     '{"decision": true, "rationale": "No more info is needed."}'),
    ("negation: 'nothing ' before 'unclear' affirms",
     '{"decision": true, "rationale": "Nothing unclear remains."}'),
    ("negation: \"n't \" before 'unclear' affirms",
     '{"decision": true, "rationale": "The request isn\'t unclear."}'),
    ("negation: 'never ' before 'insufficient' affirms",
     '{"decision": true, "rationale": "The detail was never insufficient."}'),
    ("negation window: a negator starting exactly 12 chars before the cue is inside the window",
     '{"decision": true, "rationale": "It is not so very unclear."}'),
    ("negation window: negator 13 chars before the cue is outside the 12-char window",
     '{"decision": true, "rationale": "It is not entirely unclear."}'),
    ("negation clause boundary: ';' stops a negator in the earlier clause",
     '{"decision": true, "rationale": "The user did not say; more info is needed."}'),
    ("negation clause boundary: ',' stops a negator in the earlier clause",
     '{"decision": true, "rationale": "Not so, unclear though."}'),
    ("negation clause boundary: '.' stops a negator in the earlier sentence",
     '{"decision": true, "rationale": "Not so. Unclear though."}'),
    ("negation: a later un-negated repeat of a negated cue contradicts",
     '{"decision": true, "rationale": "No more info needed, but more info would help."}'),
    ("fenced json bool true: advance", '```json\n{"decision": true, "rationale": "ok"}\n```'),
    ("prose then own-line object: advance",
     'Thinking it over.\n{"decision": true, "rationale": "The user confirmed."}\n'),
    ("own-line rule: trailing prose on the closing line is quoted, not asserted",
     'Hmm.\n{"decision": true} is what I would say, but no.\n'),
    ("own-line rule: require_key skips a keyless own-line object, the verdict advances",
     'Example:\n{"note": 1}\n{"decision": true}\n'),
    ("duplicate keys: the last value wins (advance)", '{"decision": false, "decision": true}'),
    ("duplicate keys: the last value wins (suspend)", '{"decision": true, "decision": false}'),
    ("malformed: not json at all", "not json at all"),
    ("malformed: verdict quoted mid-sentence",
     'I would answer {"decision": true} but they did not.'),
    ("malformed: two own-line verdicts disagree", 'a\n{"decision": true}\n{"decision": false}\n'),
    ("malformed: truncated object", '{"decision": true, "rationale": "cut'),
    ("malformed: top-level array", '[{"decision": true}]'),
    ("malformed: empty reply", ""),
    ("malformed: whitespace-only reply", "  \n "),
    ("malformed: non-string content (null)", None),
]


def probe_cases() -> list[tuple[str, str]]:
    """One row per `guards._NEGATION_CUES` entry, un-negated (contradicts); one per
    (`guards._NEGATORS` entry, cue) pair, negated (affirms); and, for a cue ending in a space, the
    bare word without it (not that cue). Iterates falkor-chat's real tables."""
    rows: list[tuple[str, str]] = []
    for cue in guards._NEGATION_CUES:
        word = cue.rstrip()
        rows.append((f"probe: cue {cue!r} un-negated contradicts",
                     _verdict(f"Status: {word} for now.")))
        if word != cue:
            rows.append((f"probe: {word!r} without the cue's trailing space is not cue {cue!r}",
                         _verdict(f"Status: {word}.")))
        for negator in guards._NEGATORS:
            rows.append((f"probe: negator {negator!r} before cue {cue!r} affirms",
                         _verdict(f"Status: {negator}{word} for now.")))
    return rows


# --- normalize-turns.jsonl -----------------------------------------------------------------------


def _row(name: str, text: object, role: str = "user") -> dict[str, object]:
    return {"displayName": name, "role": role, "text": text}


_KEPT = guards.RECENT_TURNS_N
_TEXT_MAX = guards.TURN_TEXT_MAX

NORMALIZE_CASES: list[tuple[str, list[dict[str, object]]]] = [
    ("empty thread", []),
    ("speaker: displayName, else authorId, else member; an empty displayName falls through",
     [{"displayName": "Ana", "authorId": "u1", "text": "a"}, {"authorId": "u2", "text": "b"},
      {"text": "c"}, {"displayName": "", "authorId": "u3", "text": "d"},
      {"displayName": "", "authorId": "", "text": "e"}]),
    ("speaker and role that are not strings are stringified",
     [{"displayName": 7, "role": 3, "text": "a"}]),
    ("role: kept when set, 'user' when missing or empty",
     [_row("A", "a", "assistant"), {"displayName": "B", "text": "b"},
      {"displayName": "C", "role": "", "text": "c"}]),
    ("text: missing, null, empty and non-string rows are dropped; whitespace-only is kept",
     [{"displayName": "A"}, _row("A", None), _row("A", ""), _row("A", 5), _row("A", ["x"]),
      _row("A", " "), _row("A", "kept")]),
    ("extra row keys are not carried",
     [{"displayName": "A", "role": "user", "text": "a", "createdAt": 1752570000000,
       "msgId": "m-1", "authorId": "u1"}]),
    (f"text cut: {_TEXT_MAX} chars kept whole, {_TEXT_MAX + 1} lose the last",
     [_row("A", "a" * (_TEXT_MAX - 1) + "b"), _row("B", "c" * _TEXT_MAX + "d")]),
    (f"exactly {_KEPT} valid rows are all kept",
     [_row("A", f"t{i}") for i in range(1, _KEPT + 1)]),
    (f"{_KEPT + 1} valid rows keep the last {_KEPT}",
     [_row("A", f"t{i}") for i in range(1, _KEPT + 2)]),
    ("filter first, then the tail: dropped rows in the tail do not shrink the window",
     [*[_row("A", f"t{i}") for i in range(1, _KEPT + 1)], _row("A", ""),
      _row("A", f"t{_KEPT + 1}"), {"displayName": "A"}, _row("A", 5)]),
    ("non-ASCII and multi-line text kept raw",
     [_row("Zoë", "linha um\nlinha dois — ação"), _row("João", "e agora?")]),
]


class _ReplyLLM:
    """Stub LLM returning one canned judge reply."""

    def __init__(self, text: str | None) -> None:
        self._text = text

    def complete(self, messages: Any) -> str | None:
        return self._text


def _dumps(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False)


def main(out: Path) -> None:
    render_rows = []
    for case, condition, understanding, turns in RENDER_CASES:
        expected = app._render_judge_user(condition, understanding, turns)
        render_rows.append({"case": case, "condition": condition, "understanding": understanding,
                            "recentTurns": turns, "expected": expected})

    parse_rows = []
    for case, reply in [*REPLY_CASES, *probe_cases()]:
        raw = app._build_llm_judge(_ReplyLLM(reply))(
            "enough info?", understanding={}, recent_turns=[], ctx={}, step_output=""
        )
        parse_failure = llm.extract_own_line_json_object(reply, require_key="decision") is None
        parse_rows.append({"case": case, "reply": reply, "expected": {
            "advance": guards._coerce_verdict(raw).decision, "parseFailure": parse_failure}})

    normalize_rows = [{"case": case, "thread": thread, "expected": guards._recent_turns(thread)}
                      for case, thread in NORMALIZE_CASES]

    cases = [r["case"] for r in (*render_rows, *parse_rows, *normalize_rows)]
    if len(cases) != len(set(cases)):
        raise SystemExit("duplicate case names")
    for name, rows in (("render-guard.jsonl", render_rows),
                       ("handoff-parse-guard.jsonl", parse_rows),
                       ("normalize-turns.jsonl", normalize_rows)):
        with (out / name).open("w", encoding="utf-8", newline="\n") as fh:
            fh.writelines(_dumps(row) + "\n" for row in rows)
        print(f"{name}: {len(rows)} rows")


if __name__ == "__main__":
    if len(sys.argv) != 2 or not Path(sys.argv[1]).is_dir():
        raise SystemExit(f"usage: {sys.argv[0]} OUT_DIR (an existing directory)")
    main(Path(sys.argv[1]))
