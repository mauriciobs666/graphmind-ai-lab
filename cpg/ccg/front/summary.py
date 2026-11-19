"""The optional plain-language summary of an answer (C-1001 P1-26 behaviours (5), (9), (10); ``docs/plans/connected-code-graph-answer-spec2.md`` §5.1).

``summarize(doc, run)`` is **aggregate-only**: the model is given counts (:func:`digest`), never a name, path or id, so it cannot learn an entity to disclose.

1. ``ccg.answer.validate(doc)`` runs first; an invalid document raises and nothing is sent or rendered.
2. One model call (``answer.summarize``, ``purpose: model-call``) through the question's :class:`ccg.front.interpret.ModelRun` (the run is the interpreter's, reused).
3. **Kept only if it passes the allow-list** (:func:`rejected_tokens`): the text is one line of letters, digits, spaces and ``. , ; : ! ? ( ) ' " - ` ``; every word is in the digest's
   own vocabulary (its keys and string values, with the singular of a plural one), the fixed function words :data:`FUNCTION_WORDS`, or a language ``answer.EXT_LANG`` mints (the notice's ``langs`` are among them); every digit run is one of the digest's integers; no word is code-shaped (``a.b``, ``a::b``). A sentence with any other word, a spelled-out number,
   a control character or a newline is dropped, and a dropped summary is not shown.
4. :func:`ccg.render.common.flow_notice_lines` follows **always**, assembled by code, whether the summary was kept, dropped or absent.

**What this guarantees and what it does not.** It guarantees the kept text is made of those words and those numbers and can carry no name, path, id, escape sequence or
extra line. It does **not** guarantee the sentence is true: the words ``confirmed`` and ``possible`` may be swapped, a count attached to the wrong noun, a claim made that the
counts do not support. The summary is optional and unscored, no evaluation measures it, and it is never described as accurate.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ccg import answer
from ccg.render.common import flow_notice_lines

COMPONENT = "answer.summarize"
PROMPT = Path(__file__).resolve().parent / "prompts" / "summary.txt"
MAX_TOKENS = 200

_WORD = re.compile(r"[^\W\d_]+")
_NUMBER = re.compile(r"\d+")
_ALLOWED_CHARS = re.compile(r"[^\W_]|[ .,;:!?()'\"`-]")           # a letter, a digit, a space or one of the listed marks; nothing else (no control character, newline, ``_``, ``/``)
_CODE_SHAPED = re.compile(r"[A-Za-z0-9][.:][A-Za-z0-9]|::|[a-z][A-Z]")
#: the fixed words a summary may use besides the digest's own and the languages
FUNCTION_WORDS = frozenset("""a an the and or but of in on to for from by with at as is are was were be been it its this that these those there has have had not no nothing none all
only also which than more most less some each both out up into over under per about total overall found show shows listed list lists affect affects affected depend depends
flow analysis answer answers language languages""".split())


@dataclass(frozen=True)
class Summary:
    """``text`` is the model's sentence when kept, else None (``dropped`` says why it is None); ``lines`` is what a renderer prints: the kept summary, then the flow notice."""
    text: str | None
    dropped: bool
    lines: tuple[str, ...]

    def render(self) -> str:
        return "\n".join(self.lines)


def digest(doc: dict) -> dict[str, Any]:
    """The aggregates the model sees: the status, the direction, counts."""
    interp = doc["interpretation"]
    out: dict[str, Any] = {"status": interp["status"]}
    if interp["status"] == "resolved":
        items = doc["items"]
        confirmed = sum(1 for i in items if i["status"] == "confirmed")
        out |= {"direction": interp["request"].get("direction", "impact"), "items": len(items), "confirmed": confirmed, "possible": len(items) - confirmed,
                "groups": len(doc["groups"]), "groupedMembers": sum(len(g["members"]) for g in doc["groups"])}
    else:
        out["options"] = len(interp.get("options", []))
    return out


def _forms(word: str) -> set[str]:
    return {word, word[:-1] if word.endswith("s") else word}


def _vocabulary(doc: dict) -> set[str]:
    """The digest's keys and string values, the languages of ``answer.EXT_LANG`` (the notice's ``langs`` are minted from them), and :data:`FUNCTION_WORDS`."""
    dg = digest(doc)
    words = {w.lower() for v in (*dg, *(v for v in dg.values() if isinstance(v, str))) for w in _WORD.findall(v)}
    vocab = set(FUNCTION_WORDS)
    for w in words:
        vocab |= _forms(w)
    vocab |= set(answer.EXT_LANG.values())
    return vocab


def rejected_tokens(text: str, doc: dict) -> list[str]:
    """What keeps ``text`` from being kept: each disallowed character (as ``\\xNN``-style escapes), code-shaped token, word outside the vocabulary and number the digest does
    not hold (see the module docstring); ``[]`` when the text passes. A passing text is still not known to be true."""
    vocab = _vocabulary(doc)
    numbers = {str(v) for v in digest(doc).values() if type(v) is int}
    out = [ascii(c)[1:-1] for c in dict.fromkeys(text) if not _ALLOWED_CHARS.fullmatch(c)]
    out += [m.group() for m in _CODE_SHAPED.finditer(text)]
    out += [w for w in _WORD.findall(text) if w.lower() not in vocab]
    out += [n for n in _NUMBER.findall(text) if n not in numbers]
    return out


def summarize(doc: dict, run) -> Summary:
    """The summary lines of ``doc``. Raises ``AnswerError`` for an invalid document (before any call) and ``ccg.front.errors.ModelError`` when the model cannot be reached."""
    answer.validate(doc)
    dg = digest(doc)
    body = {"messages": [{"role": "system", "content": PROMPT.read_text(encoding="utf-8")}, {"role": "user", "content": json.dumps(dg, sort_keys=True)}],
            "temperature": 0, "max_tokens": MAX_TOKENS, "stream": False}
    model = run.bound_model(COMPONENT)
    body["model"] = model["id"]
    text = run.complete(COMPONENT, body).strip()
    keep = bool(text) and not rejected_tokens(text, doc)
    notice = tuple(flow_notice_lines(doc))
    return Summary(text if keep else None, not keep, ((text,) if keep else ()) + notice)
