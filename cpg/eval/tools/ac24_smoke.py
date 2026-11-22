#!/usr/bin/env python3
"""ac24_smoke.py — the AC-24 smoke harness (P1-35; ``docs/plans/connected-code-graph-p1-gates.md`` P1-35).

Spec: ``docs/plans/connected-code-graph-p1-ml.md`` §4.1-§4.3 (protocol, ``ccg-prereg-ac24-v1`` contents, custody §6.4); eval protocol
``docs/plans/connected-code-graph-eval-protocol-ml.md`` §10.4 (scorer), §10.5 (thresholds), §12.3 (record schema); answer-spec2 §2 (``cacheBypass``).
Reviews: ``docs/reviews/connected-code-graph-p1-35-impl.md`` (analyst), ``-p1-35-ml.md`` (data-scientist).

Verbs (stdout: counts and the output path only; qids go to the output JSON, never question text):
  dev       [--dev-items F] [--system S] [--out F] [--prereg F]
              the dev entry check: the 14 dev items, repeat 1 only, cache on and empty at the start, after the canary. Passes at >= 11/12 unambiguous
              and 2/2 ambiguous; a failure, or more than 1 of the 14 replies unusable, reports "entry check failed: do not edit the prompt, grammar or resolver after dev2 exists; held-out items are never used for tuning".
              Writes the run record (default ``cpg/eval/results/ac24-dev-run.json``). A ``--dev-items`` path under the custodian area is refused.
  canary    [--system S]
              runs the fixed made-up ``fx-*`` questions (``CANARY_QUESTIONS``) through the rig and prints the sha256 digest of the model's replies.
  tree-hashes
              prints ``coreTreeHash``, ``packSetHash``, ``harnessHash``, HEAD and the tree's cleanliness (what the held-out pre-flight recomputes).
  heldout   --heldout [--custodian-root D] [--prereg F] --dev-result F [--system S] [--after-adaptation]
              the held-out run, once. Pre-flight, all before the once-only marker and none of it reading item text: the explicit flag; the
              ``ccg-prereg-ac24-v1`` record in the file and in ``HEAD``'s copy; the dev result passed with the record's interpreter and canary; the held-out
              list's sha256 equals the record's and ``heldout-manifest.json``'s; ``coreTreeHash``, ``packSetHash``, ``harnessHash`` recomputed from the working tree,
              a clean tree under ``cpg/{ccg,clients,eval/tools,packs}`` and the record's ``coreCommit`` an ancestor of HEAD; the rig is built; the cache is empty; the effective
              sampler (a non-null record value with no observer is a refusal); the canary digest equals the record's. A difference in the tree or the harness refuses the
              run unless ``--after-adaptation``, which stamps the result ``afterAdaptation`` with the reasons. Then the marker (bound to the EVAL_COMMIT: one file in
              the DERIVED custodian root, ``ac24-heldout-once-<E12>.json``; a ``--custodian-root`` other than the derived one is refused), then the items, which
              are read only from ``<custodian root>/heldout/ac24.jsonl``, re-counted (24 unambiguous 12/12, 6 ambiguous) and run. An existing marker refuses the run
              whatever its state; a human removes it after a harness error.
  prereg-record --dev-result F --eval-commit SHA --core-commit SHA [--heldout-list-sha256 H] [--out F]
              prints the ``ccg-prereg-ac24-v1`` record built from the final dev run (it NEVER appends it, and refuses an ``--out`` that exists). The three hashes
              come only from the dev result's ``treeStamp``; it refuses unless the current tree equals the stamp and both are clean (N1). There are no hash flags.

Scoring (§10.4, mechanical): an unambiguous item is understood iff ``status == resolved`` and the request equals gold (symbol anchors compared as ``E(qname)``,
EP §10.4's E1, from ``match_keys``); an ambiguous item iff ``status != resolved`` with ``goldOptions == []``, else ``status == ambiguous`` with every gold
option among ``options``, and always ``|options| <= CAP_i = max(capFloor, |goldOptions_i|)``. **Repeat 1 is the scored draw** (cache on and asserted empty,
``cacheHit`` must be false: true scores the item "not understood" and stops the run, C5); repeats 2-5 run with ``cacheBypass`` and give the flip rate and
per-draw rate, reported only. A ``cacheHit`` on a repeat 2-5 voids that repeat. The endpoint count is of POST completions only (never the ``GET /api/v0/models``
read): at least 1 per interpreter repeat, every extra one a logged retry (same body after a transport error or a non-2xx).

**The unusable-reply observable is recorded, not inferred.** ``Front.answer`` reads a reply that is not JSON or does not validate as ``Interpretation.draft is
None``. ``recording_interpreter`` subclasses ``Interpreter`` and appends, per ``interpret`` call, ``{draftNone, provenance}`` to ``RequestLog.events``; the run
scores "unusable, not understood" from that record on every item. A valid draft that names nothing is a usable reply and is scored by EP §10.4 like any other
(a ``no-anchor`` ask is understood on a ``goldOptions == []`` item). The provenance of the event also serves an outcome the front refused as not built
after the model ran, so the fingerprint and sampler checks cover it.

Parser-resolved items (no ``provenance``, no model call) are tallied apart: not interpreter draws, one repeat, no flip rate; they still enter the pooled k/n,
so the report shows the pooled composition (parser-resolved count and the interpreter-only rate). The harness reports Wilson intervals and **never claims the
run certifies 0.92**.

Exit: 0 pass, 1 the gate or the dev entry check failed (or the run stopped, C5), 2 usage, 4 a refusal (guard, missing or invalid input), 5 a harness error (an
assertion about the run itself failed; no score is produced).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_CPG = _HERE.parents[1]
for _p in (str(_CPG), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from match_keys import E as eligible  # noqa: E402  (EP §2.3: the eligible ancestor of a qname)

PREREG_SCHEMA = "ccg-prereg-ac24-v1"
CHAT_PATH = "/v1/chat/completions"
REPEATS_HELDOUT = 5
CAP_FLOOR = 5
CAP_RULE = "max(capFloor, |goldOptions_i|)"
CACHE_BYPASS_NAME = "cacheBypass"
POOLED_RATE = Fraction(92, 100)           # EP §10.5
LANG_RATE = Fraction(85, 100)
LANGS = ("en", "pt")
#: what the held-out split must contain (EP §10.2/§10.5; P1-35 ledger counts); a difference means the method owner recomputes the thresholds before the run
EXPECTED_HELDOUT = {"unambiguous": 24, "en": 12, "pt": 12, "ambiguous": 6}
DEV_UNAMBIGUOUS_MIN = 11                  # of 12
INTERPRETER_KEYS = ("endpoint", "modelFingerprint", "quantization", "promptHash", "grammarHash", "samplingHash", "resolverVersion")
CLAIM = "smoke result: shows the interpreter is not broken on the impact cells; it does not certify 0.92"
#: the directories whose uncommitted state refuses a held-out run (``cpg/eval/tools`` holds this harness; ``cpg/packs`` is hashed by ``packSetHash``)
CLEAN_PATHS = ("cpg/ccg", "cpg/clients", "cpg/eval/tools", "cpg/packs")
CORE_EXCLUDED_TOP = ("eval", "packs", "tests")        # EP §12.3: coreTreeHash excludes cpg/eval/**, cpg/packs/**, cpg/tests/** and **/*.md
REPEAT_PROTOCOL = {"repeats": 5, "repeat1": "no cacheBypass, cache asserted empty before the run, scored, cacheHit must be false (C5)",
                   "repeats2to5": "cacheBypass, reported only (flip rate, per-draw rate)"}
UNUSABLE_OBSERVABLE = "Interpretation.draft is None, recorded per interpret call by recording_interpreter (not inferred from the answer's shape)"
#: made-up questions over ``fx-*`` names; the canary's replies are digested, never scored, and nothing here is derived from the dev or held-out items
CANARY_QUESTIONS = (
    "If I rename the field retries in fx-billing/ledger.py::FxLedger.post, what else breaks?",
    "o que depende de fx-billing/ledger.py::FxLedger.post no resto do sistema?",
    "tell me about fx-orders/cart.ts::FxCart",
    "which parts of the code does fx-orders/cart.ts::FxCart.add itself call or use?",
    "POST /fx/orders/{order_id}/refund quem consome essa rota?",
    "what is it",
)

#: ``-p1-ml.md`` §4.2 "Resolver rules as built", as recorded in ``resolverRules`` (written from that section at P1-35)
RESOLVER_RULES = (
    "option order: candidate-major, impact before depends-on, ranked 1..; whole candidates beyond OPTIONS_MAX=20 dropped, never cut mid-candidate",
    "CAP=5 applies to did-you-mean lists only (edit distance to the last dotted segment, then qname); complete readings are uncapped",
    "path-suffix join: '/'-bounded suffix read only after every exact read misses (symbol+file and file+line); several hits are ambiguous no-anchor, never a pick; "
    "a path held as skipped never falls back to the suffix read",
    "unfound name beside a resolving group: a name-bearing group (kind symbol, route, tool or line) that finds nothing makes the answer ambiguous "
    "(unknown-name with did-you-mean candidates, else no-anchor); groups of only other/server spans, or a plain file with no anchor form, are ignored; "
    "a failed file path rides as lookups, not as an ask",
    "ambiguous vs unresolved: unresolved only for a failed named path (with lookups) or a file held only as skipped; no anchor named is ambiguous",
    "the parser claims only the literal direction tokens 'impact' and 'depends-on'; 'depends on' is left to the interpreter",
    "a parser-resolved answer carries no provenance; cacheBypass is a no-op on it",
)
DIRECTION_CUES = {
    "neutral (direction null)": ["an overview of", "o panorama de"],
    "depends-on": ["what the anchor relies on", "requires", "is built on", "itself calls or uses"],
    "impact": ["a passive 'what calls X'"],
}
UNBUILT_READINGS_RULE = ("a draft whose use case or any readings use case is unbuilt is refused whole (typed not-built, request echoed), never answered for its "
                         "built half; the harness scores a not-built outcome 'not understood' on a built-use-case item; dev gold has no unbuilt use case (0/14)")


class HarnessError(RuntimeError):
    """An assertion about the run itself failed (fingerprint, sampler, cache, endpoint count): no score is produced."""


class Refusal(RuntimeError):
    """A guard refused the run before it started (exit 4)."""


# --------------------------------------------------------------------------------------------------------------- statistics

def wilson(k: int, n: int, z: float = 1.959964) -> list[float] | None:
    """The 95% Wilson interval of k/n, rounded to 3 places; None for n == 0."""
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)]


def threshold(rate: Fraction, n: int) -> int:
    """``ceil(rate * n)`` in exact arithmetic (EP §10.5: ceil(0.92 x 24) = 23, ceil(0.85 x 12) = 11)."""
    return -((-rate.numerator * n) // rate.denominator)


def rate_block(k: int, n: int) -> dict[str, Any]:
    return {"k": k, "n": n, "wilson95": wilson(k, n)}


# --------------------------------------------------------------------------------------------------------------- items

def load_items(path: Path) -> list[dict[str, Any]]:
    items = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            it = json.loads(line)
        except ValueError:
            raise Refusal(f"{path}: line {i} is not JSON") from None
        if not isinstance(it, dict):
            raise Refusal(f"{path}: line {i} is not an object")
        for k in ("qid", "text", "lang", "outcome"):
            if not isinstance(it.get(k), str) or not it[k]:
                raise Refusal(f"{path}: line {i} has no {k}")
        if it["outcome"] == "unambiguous":
            if not isinstance(it.get("gold"), dict):
                raise Refusal(f"{path}: line {i} (unambiguous) has no gold")
        elif it["outcome"] == "ambiguity":
            if not isinstance(it.get("goldOptions"), list):
                raise Refusal(f"{path}: line {i} (ambiguity) has no goldOptions")
        else:
            raise Refusal(f"{path}: line {i} has an unknown outcome")
        items.append(it)
    if len({it["qid"] for it in items}) != len(items):
        raise Refusal(f"{path}: duplicate qids")
    return sorted(items, key=lambda it: it["qid"])


def recount(items: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """The harness re-counts the unambiguous and ambiguous items (and the unambiguous by language) from the split output itself."""
    c = {"unambiguous": 0, "ambiguous": 0, **{lang: 0 for lang in LANGS}}
    for it in items:
        if it["outcome"] == "unambiguous":
            c["unambiguous"] += 1
            c[it["lang"]] = c.get(it["lang"], 0) + 1
        else:
            c["ambiguous"] += 1
    return c


# --------------------------------------------------------------------------------------------------------------- scorer (EP §10.4)

def _anchor_key(a: Mapping[str, Any]) -> tuple:
    """An anchor's identity without its engine id. E1 (EP §10.4): a symbol anchor is compared as ``E(qname)``, the eligible ancestor, so a lambda enclosing a
    line scores as its enclosing element. Known limit (EP §10.4, P2-2): two ``#k`` duplicates share one ``E``."""
    t = a.get("type")
    if t == "symbol":
        q = a.get("qname")
        return ("symbol", a.get("repoId"), eligible(q) if isinstance(q, str) else q)
    if t == "contract":
        return ("contract", a.get("kind"), a.get("keyNorm"))
    if t == "config":
        return ("config", a.get("repoId"), a.get("path"), a.get("keyPath"))
    return ("other", json.dumps(a, sort_keys=True))


def request_key(r: Mapping[str, Any]) -> tuple:
    """useCase, direction, the anchor set, the scope (``null`` is the whole system); ``rank`` is not part of an interpretation."""
    scope = r.get("scope")
    return (r.get("useCase"), r.get("direction"), frozenset(_anchor_key(a) for a in r.get("anchors", [])),
            json.dumps(scope, sort_keys=True) if scope else None)


def score_item(item: Mapping[str, Any], doc: Mapping[str, Any] | None, cap_floor: int = CAP_FLOOR, *, unusable: bool = False) -> dict[str, Any]:
    """The verdict of one answer (``doc`` None: the front refused it as not built). ``unusable`` is the recorded observable (``Interpretation.draft is None``).
    Never reads ``item['text']``."""
    v: dict[str, Any] = {"understood": False, "reason": None, "status": None, "nOptions": None, "extras": None, "capExceeded": False, "overAsk": False,
                         "unusable": bool(unusable), "notBuilt": doc is None, "ambiguityKindEmitted": None}
    ambiguous_item = item["outcome"] == "ambiguity"
    if doc is None:
        v["reason"] = "not-built"
        return v
    interp = doc["interpretation"]
    status = interp["status"]
    v["status"] = status
    v["ambiguityKindEmitted"] = interp.get("ambiguityKind")
    options = interp.get("options") or []
    v["nOptions"] = len(options)
    if unusable:
        v["reason"] = "unusable-reply"
        v["overAsk"] = not ambiguous_item
        return v
    if not ambiguous_item:
        if status == "resolved":
            ok = request_key(interp["request"]) == request_key(item["gold"])
            v["understood"], v["reason"] = ok, "match" if ok else "wrong-interpretation"
        else:
            v["overAsk"], v["reason"] = True, "over-asking"
        return v
    gold = item["goldOptions"]
    cap = max(cap_floor, len(gold))
    if status == "resolved":
        v["reason"] = "silent-single-interpretation"
        return v
    if gold:
        have = {request_key(o) for o in options}
        matched = sum(1 for g in gold if request_key(g) in have)
        v["extras"] = len(options) - matched
        if status != "ambiguous":
            v["reason"] = "unresolved-names-no-reading"
            return v
        if matched != len(gold):
            v["reason"] = "gold-option-missing"
            return v
    else:
        v["extras"] = len(options)
    if len(options) > cap:
        v["capExceeded"], v["reason"] = True, "cap-exceeded"
        return v
    v["understood"], v["reason"] = True, "asked"
    return v


# --------------------------------------------------------------------------------------------------------------- endpoint log and draft record

class RequestLog:
    """What the interpreter endpoint was asked, as the broker seam saw it (``entries``: ``{method, path, body, status, reply?}``; status None: no completed
    exchange), and what the interpreter returned (``events``: one ``{draftNone, provenance}`` per ``interpret`` call)."""

    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []

    def mark(self) -> tuple[int, int]:
        return (len(self.entries), len(self.events))

    def since(self, mark: tuple[int, int]) -> list[dict[str, Any]]:
        return self.entries[mark[0]:]

    def events_since(self, mark: tuple[int, int]) -> list[dict[str, Any]]:
        return self.events[mark[1]:]


def _reply_text(resp: Any) -> str | None:
    """The message content of a completion response (the stable part: the envelope carries ids and timings)."""
    try:
        return json.loads(resp.body)["choices"][0]["message"]["content"]
    except Exception:  # noqa: BLE001 - a response that is not a completion has no reply
        return None


class CountingBroker:
    """A counting proxy over a ``BrokerClient`` (or its double): delegates ``open_run``/``close_run``/``request`` and logs every request, with the completion's reply
    text. It counts at the broker seam (one broker request is one endpoint request; the interpreter's own retry re-enters ``request``), not in the endpoint's log."""

    def __init__(self, inner: Any, log: RequestLog | None = None):
        self._inner = inner
        self.log = log if log is not None else RequestLog()

    def open_run(self, *a, **k):
        return self._inner.open_run(*a, **k)

    def close_run(self, token):
        return self._inner.close_run(token)

    def request(self, token, request, body=None):
        entry = {"method": request["target"]["method"], "path": request["target"]["path"], "body": body, "status": None}
        self.log.entries.append(entry)
        decision, resp = self._inner.request(token, request, body)
        entry["status"] = None if resp is None else resp.status
        if resp is not None and entry["method"] == "POST" and entry["path"] == CHAT_PATH:
            entry["reply"] = _reply_text(resp)
        return decision, resp


def draft_record(draft: Any) -> dict[str, Any] | None:
    """The draft as the interpreter returned it, in the reply's own shape (``useCase``, ``direction``, ``anchors`` groups of ``{text, kindHint}`` spans, ``scope``,
    ``readings``); None for an unusable reply. Dev items only: it carries verbatim question text, so it is never written for a held-out run."""
    if draft is None:
        return None
    return {"useCase": draft.use_case, "direction": draft.direction,
            "anchors": [{"spans": [{"text": sp.text, "kindHint": sp.kind_hint} for sp in g.spans]} for g in draft.anchors],
            "scope": draft.scope, "readings": [{"useCase": r.use_case, "direction": r.direction} for r in draft.readings]}


def recording_interpreter(log: RequestLog, cache: Any = None):
    """An ``Interpreter`` that records, per ``interpret`` call, whether the reply was unusable (``draft is None``), the draft itself (:func:`draft_record`) and the
    call's provenance, into ``log.events``."""
    from ccg.front import interpret

    class Recording(interpret.Interpreter):
        def interpret(self, question, snapshot_id, run, *, bypass=False):
            result = super().interpret(question, snapshot_id, run, bypass=bypass)
            log.events.append({"draftNone": result.draft is None, "draft": draft_record(result.draft), "provenance": dict(result.provenance)})
            return result
    return Recording(cache)


def completions(entries: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """POST completions only: the ``GET /api/v0/models`` read is not a completion."""
    return [e for e in entries if e["method"] == "POST" and e["path"] == CHAT_PATH]


def check_completion_count(entries: Sequence[Mapping[str, Any]]) -> int:
    """At least 1 completion; every one after the first a logged retry (the same body as the one before it, which got no completed exchange or a non-2xx).
    Returns the count; raises :class:`HarnessError`."""
    posts = completions(entries)
    if not posts:
        raise HarnessError("an interpreter repeat sent no completion request to the endpoint (cacheHit is the engine's self-report)")
    for prev, cur in zip(posts, posts[1:]):
        failed = prev["status"] is None or not 200 <= prev["status"] < 300
        if not (failed and cur["body"] == prev["body"]):
            raise HarnessError("an extra completion request is not a retry of a failed one (a different body, or the first exchange succeeded)")
    return len(posts)


# --------------------------------------------------------------------------------------------------------------- the canary

def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def run_canary(client: Any, log: RequestLog) -> dict[str, Any]:
    """The fixed ``fx-*`` questions through the rig (cache bypassed, so the cache stays empty), and the sha256 of the model's reply text per question. Detects drift in
    preset, template or model without reading the server's configuration. Every question must reach the model."""
    from clients.common import ClientError, ModelUnavailable, NotBuilt
    replies = []
    for q in CANARY_QUESTIONS:
        mark = log.mark()
        try:
            client.ask(question=q, cache_bypass=True)
        except NotBuilt:
            pass
        except ModelUnavailable as exc:
            raise HarnessError(f"the interpreter model was unavailable during the canary ({exc.code})") from None
        except ClientError as exc:
            raise HarnessError(f"the client failed on a canary question ({exc.code})") from None
        posts = completions(log.since(mark))
        if not posts:
            raise HarnessError("a canary question did not reach the model")
        replies.append(posts[-1].get("reply"))
    qsha = sha256_text(canonical_json(list(CANARY_QUESTIONS)))
    return {"questionsSha256": qsha, "digest": sha256_text(canonical_json({"questions": qsha, "replies": replies})), "n": len(replies)}


# --------------------------------------------------------------------------------------------------------------- the run

def _provenance_mismatch(prov: Mapping[str, Any], expected: Mapping[str, Any]) -> list[str]:
    return [k for k in INTERPRETER_KEYS if prov.get(k) != expected.get(k)]


def check_sampler_effective(sampler_observer: Callable[[], Mapping[str, Any]] | None, expected: Mapping[str, Any] | None) -> dict[str, Any] | None:
    effective = dict(sampler_observer()) if sampler_observer is not None else None
    if expected is not None and effective != dict(expected):
        raise HarnessError("the effective sampler configuration differs from the pre-registered one")
    return effective


def run_items(items: Sequence[Mapping[str, Any]], client: Any, *, repeats: int, log: RequestLog, cache_len: Callable[[], int],
              expected_interpreter: Mapping[str, Any] | None = None, expected_sampler_request: Mapping[str, Any] | None = None,
              sampler_observer: Callable[[], Mapping[str, Any]] | None = None, expected_sampler_effective: Mapping[str, Any] | None = None,
              cap_floor: int = CAP_FLOOR, record_drafts: bool = False) -> dict[str, Any]:
    """Drive ``client.ask(question=, cache_bypass=)`` (a ``clients.common.Client`` over the real ``Front``) over ``items`` in qid order, item by item.

    Repeat 1 is not bypassed (the cache must be empty at the start of the run and ``cacheHit`` false); repeats 2..``repeats`` pass ``cache_bypass=True``.
    Returns the run record: per-item observations and the stop reason, if any. Raises :class:`HarnessError` for any failed assertion about the run itself.
    ``record_drafts`` (dev only; the held-out run never sets it: a draft is verbatim question text) stores the observed draft on each repeat as ``draft`` (a field
    no hash and no verdict reads)."""
    from clients.common import ClientError, ModelUnavailable, NotBuilt

    if cache_len() != 0:
        raise HarnessError("the interpretation cache is not empty before repeat 1 (a fresh cache per gate run)")
    effective = check_sampler_effective(sampler_observer, expected_sampler_effective)
    baseline = dict(expected_interpreter) if expected_interpreter is not None else None
    out_items, observed_prov, stop = [], None, None
    for item in sorted(items, key=lambda it: it["qid"]):
        rec: dict[str, Any] = {"qid": item["qid"], "lang": item["lang"], "outcome": item["outcome"], "modelCalled": None, "parserResolved": False, "repeats": []}
        n_repeats = repeats
        r = 1
        while r <= n_repeats:
            bypass = r > 1
            mark = log.mark()
            try:
                doc, err = client.ask(question=item["text"], cache_bypass=bypass), None
            except NotBuilt:
                doc, err = None, "not-built"
            except ModelUnavailable as exc:
                raise HarnessError(f"the interpreter model was unavailable ({exc.code}); nothing to score") from None
            except ClientError as exc:
                raise HarnessError(f"the client failed on a question ({exc.code})") from None
            entries = log.since(mark)
            posts = completions(entries)
            evs = log.events_since(mark)
            ev = evs[-1] if evs else None
            doc_prov = doc["interpretation"].get("provenance") if doc is not None else None
            if doc is not None and doc_prov is None and (posts or ev):
                raise HarnessError("a completion request was sent but the answer carries no provenance")
            prov = doc_prov or (ev["provenance"] if ev else None)       # an outcome refused as not built after the model ran still has the call's provenance
            if r == 1:
                rec["modelCalled"] = prov is not None or bool(posts)
                if not rec["modelCalled"]:
                    rec["parserResolved"] = doc is not None
                    n_repeats = 1                                # no model call: one repeat, no flip rate
            obs: dict[str, Any] = {"repeat": r, "cacheBypass": bypass, "error": err, "completions": len(posts), "voided": False, "cacheHit": None}
            if prov is not None:
                # the unit of observation: the call's own provenance, compared with the pre-registered interpreter block (a mismatch is not a score)
                if baseline is None:
                    baseline = {k: prov.get(k) for k in INTERPRETER_KEYS}
                bad = _provenance_mismatch(prov, baseline)
                if bad:
                    raise HarnessError(f"the provenance differs from the interpreter block in {bad}")
                observed_prov = {k: prov.get(k) for k in INTERPRETER_KEYS}
                obs["cacheHit"] = prov["cacheHit"]
                if bypass and not prov["cacheHit"] and prov.get("cacheBypass") is not True:   # a hit on a bypassed repeat is voided below (the schema forbids both)
                    raise HarnessError("a repeat asked with the cache bypass came back without provenance.cacheBypass (the input did not reach the engine)")
                if not bypass and "cacheBypass" in prov:
                    raise HarnessError("repeat 1 carries provenance.cacheBypass")
                if not prov["cacheHit"]:                         # a cache hit is served without a completion; a "false" must be backed by the endpoint's own count
                    obs["completions"] = check_completion_count(entries)
                elif posts:
                    raise HarnessError("cacheHit is true but a completion request was sent")
            if (posts or prov is not None) and ev is None and not (prov is not None and prov["cacheHit"]):
                raise HarnessError("a model call left no draft observation (the rig's interpreter is not the recording one)")
            if posts and expected_sampler_request is not None:
                for e in posts:
                    body = e["body"] or {}
                    if {k: body.get(k) for k in expected_sampler_request} != dict(expected_sampler_request):
                        raise HarnessError("a completion request's sampler parameters differ from the pre-registered request body")
            unusable = bool(ev and ev["draftNone"])
            verdict = score_item(item, doc, cap_floor, unusable=unusable)
            obs.update(verdict=verdict, understood=verdict["understood"], **({"draft": ev.get("draft") if ev else None} if record_drafts else {}))
            if prov is not None and prov["cacheHit"]:
                if r == 1:
                    obs["understood"] = verdict["understood"] = False
                    verdict["reason"] = "c5-cache-hit-on-repeat-1"
                    rec["repeats"].append(obs)
                    stop = {"code": "C5", "qid": item["qid"], "why": "cacheHit true on repeat 1: the cache was not empty for the gate snapshot"}
                    break
                obs["voided"] = True
            rec["repeats"].append(obs)
            r += 1
        out_items.append(rec)
        if stop:
            break
    return {"items": out_items, "stop": stop, "ran": len(out_items), "of": len(items), "interpreterObserved": observed_prov, "samplerEffective": effective,
            "repeatsPlanned": repeats, "order": "qid-sorted, item by item", "capFloor": cap_floor}


# --------------------------------------------------------------------------------------------------------------- the verdict

def _first(rec: Mapping[str, Any]) -> Mapping[str, Any]:
    return rec["repeats"][0]


def summarize(run: Mapping[str, Any], items: Sequence[Mapping[str, Any]], *, mode: str) -> dict[str, Any]:
    """Thresholds and reports from the re-counted items and the run's repeat-1 verdicts. ``mode``: ``dev`` or ``heldout``."""
    counts = recount(items)
    scored = [rec for rec in run["items"] if rec["repeats"]]
    unamb = [r for r in scored if r["outcome"] == "unambiguous"]
    amb = [r for r in scored if r["outcome"] == "ambiguity"]
    k_unamb = sum(1 for r in unamb if _first(r)["understood"])
    k_amb = sum(1 for r in amb if _first(r)["understood"])
    per_lang = {lang: (sum(1 for r in unamb if r["lang"] == lang and _first(r)["understood"]), counts.get(lang, 0)) for lang in LANGS}
    over = [r["qid"] for r in unamb if _first(r)["verdict"]["overAsk"]]
    unusable = [r["qid"] for r in scored if _first(r)["verdict"]["unusable"]]
    # interpreter items: draws, flips, per-draw rate; parser-resolved and no-model items are tallied apart
    interp_items = [r for r in scored if r["modelCalled"] and not r["parserResolved"]]
    draws = [[x["understood"] for x in r["repeats"] if not x["voided"]] for r in interp_items]
    flips = sum(1 for d in draws if len(d) >= 2 and len(set(d)) > 1)
    flip_n = sum(1 for d in draws if len(d) >= 2)
    per_draw_k = sum(sum(d) for d in draws)
    per_draw_n = sum(len(d) for d in draws)
    parser_unamb = [r for r in unamb if r["parserResolved"]]
    k_interp = sum(1 for r in unamb if not r["parserResolved"] and _first(r)["understood"])
    extras = {r["qid"]: {"nOptions": _first(r)["verdict"]["nOptions"], "extras": _first(r)["verdict"]["extras"],
                         "emittedKind": _first(r)["verdict"]["ambiguityKindEmitted"], "capExceeded": _first(r)["verdict"]["capExceeded"]} for r in amb}
    s: dict[str, Any] = {
        "mode": mode, "claim": CLAIM, "recount": counts, "ran": run["ran"], "of": run["of"], "stop": run["stop"],
        "unambiguousUnderstood": rate_block(k_unamb, counts["unambiguous"]),
        "pooledComposition": {"parserResolvedUnambiguous": len(parser_unamb), "interpreterDrawsUnambiguous": len(unamb) - len(parser_unamb),
                              "interpreterOnlyUnderstood": rate_block(k_interp, len(unamb) - len(parser_unamb)),
                              "note": "the pooled k/n mixes parser-resolved and interpreter draws; the interpreter-only rate is beside it"},
        "ambiguousUnderstood": rate_block(k_amb, counts["ambiguous"]),
        "perLanguage": {lang: rate_block(k, n) for lang, (k, n) in per_lang.items()},
        "overAsking": rate_block(len(over), counts["unambiguous"]), "overAskedQids": over,
        "ambiguousExtras": extras, "capExceededQids": [q for q, e in extras.items() if e["capExceeded"]],
        "unusableObserved": {"count": len(unusable), "qids": unusable, "of": len(scored), "observable": UNUSABLE_OBSERVABLE},
        "notBuilt": [r["qid"] for r in scored if _first(r)["verdict"]["notBuilt"]],
        "parserResolved": {"count": sum(1 for r in scored if r["parserResolved"]), "qids": [r["qid"] for r in scored if r["parserResolved"]],
                           "note": "no provenance, no model call: not interpreter draws, one repeat, no flip rate"},
        "interpreterItems": len(interp_items), "flipRate": rate_block(flips, flip_n),
        "perDrawUnderstood": rate_block(per_draw_k, per_draw_n),
        "voidedRepeats": sum(1 for r in scored for x in r["repeats"] if x["voided"]),
        "completionsByItem": {r["qid"]: [x["completions"] for x in r["repeats"]] for r in scored},
    }
    if mode == "dev":
        ok = k_unamb >= DEV_UNAMBIGUOUS_MIN and k_amb == counts["ambiguous"] and counts["ambiguous"] > 0
        s["thresholds"] = {"unambiguousMin": DEV_UNAMBIGUOUS_MIN, "ambiguousAll": counts["ambiguous"]}
        s["entryPass"] = bool(ok and not run["stop"] and run["ran"] == run["of"] and len(unusable) <= 1)
        if not s["entryPass"]:
            s["recommendation"] = "entry check failed: do not edit the prompt, grammar or resolver after dev2 exists; held-out items are never used for tuning"
        s["verdict"] = "dev entry check passed" if s["entryPass"] else "dev entry check failed"
        return s
    pooled_min = threshold(POOLED_RATE, counts["unambiguous"])
    lang_min = {lang: threshold(LANG_RATE, counts.get(lang, 0)) for lang in LANGS}
    complete = not run["stop"] and run["ran"] == run["of"]
    s["thresholds"] = {"pooledMin": pooled_min, "perLanguageMin": lang_min, "ambiguousAll": counts["ambiguous"], "rule": "ceil(0.92 n) pooled, ceil(0.85 n) per language"}
    s["checks"] = {"pooled": k_unamb >= pooled_min, "perLanguage": {lang: per_lang[lang][0] >= lang_min[lang] for lang in LANGS},
                   "ambiguousAll": k_amb == counts["ambiguous"], "complete": complete}
    s["passed"] = bool(s["checks"]["pooled"] and all(s["checks"]["perLanguage"].values()) and s["checks"]["ambiguousAll"] and complete)
    s["verdict"] = "smoke gate passed" if s["passed"] else ("smoke gate stopped (C5)" if run["stop"] else "smoke gate failed, reported as run")
    return s


# --------------------------------------------------------------------------------------------------------------- the tree (EP §12.3)

def _git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(["git", *args], cwd=cwd or _CPG.parent, capture_output=True, text=True, check=True).stdout


def _canon_hash(pairs: Sequence[Sequence[str]]) -> str:
    """SHA-256 of the canonical JSON (bundle spec §3.1: sorted keys, ``,``/``:``, non-ASCII literal) of the array of ``[relpath, sha256]`` sorted by path."""
    body = json.dumps(sorted([list(p) for p in pairs]), ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def is_core_file(rel: str) -> bool:
    """``rel`` is relative to ``cpg/``: in ``coreTreeHash`` unless under ``eval/``, ``packs/``, ``tests/`` or a ``.md`` file (EP §12.3)."""
    return rel.split("/", 1)[0] not in CORE_EXCLUDED_TOP and not rel.endswith(".md")


def tracked_files(repo: Path | None = None) -> list[str]:
    """The git-tracked files under ``cpg/``, relative to ``cpg/`` (read-only ``git ls-files``)."""
    repo = repo or _CPG.parent
    out = _git("ls-files", "-z", "--", "cpg", cwd=repo)
    return sorted(p[len("cpg/"):] for p in out.split("\0") if p.startswith("cpg/"))


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def core_tree_hash(files: Sequence[str], cpg_dir: Path) -> str:
    """``coreTreeHash``: over the tracked files under ``cpg/`` minus the exclusions, ``relpath`` relative to ``cpg/``, bytes from the working tree."""
    return _canon_hash([(f, _file_sha(cpg_dir / f)) for f in files if is_core_file(f) and (cpg_dir / f).is_file()])


def pack_set_hash(files: Sequence[str], cpg_dir: Path) -> str:
    """``packSetHash``: over every tracked file under ``cpg/packs/``, ``relpath`` relative to ``cpg/packs/``."""
    return _canon_hash([(f[len("packs/"):], _file_sha(cpg_dir / f)) for f in files if f.startswith("packs/") and (cpg_dir / f).is_file()])


def harness_hash() -> str:
    return _file_sha(Path(__file__).resolve())


def observe_tree(core_commit: str | None = None, repo: Path | None = None) -> dict[str, Any]:
    """What the held-out pre-flight recomputes from the working tree (read-only git): the two hashes, HEAD, the dirty paths under :data:`CLEAN_PATHS`, and whether
    the record's ``coreCommit`` is an ancestor of HEAD (the record's own commit follows ``coreCommit``, so HEAD cannot equal it). ``dirty`` lists the tracked or
    untracked changes (``git status --porcelain``) and, prefixed ``!! ``, every git-IGNORED file there except bytecode under ``__pycache__``: an ignored file the
    process could import or load is not hashed, so it makes the tree not clean (N3). ``repo`` is the repository root (default: this one; tests pass a throwaway)."""
    repo = repo or _CPG.parent
    cpg_dir = repo / "cpg"
    files = tracked_files(repo)
    dirty = [ln for ln in _git("status", "--porcelain", "--", *CLEAN_PATHS, cwd=repo).splitlines() if ln.strip()]
    ignored = _git("ls-files", "-z", "--others", "--ignored", "--exclude-standard", "--", *CLEAN_PATHS, cwd=repo)
    dirty += [f"!! {p}" for p in sorted(ignored.split("\0")) if p and "__pycache__" not in p.split("/")]
    anc = None
    if core_commit:
        anc = subprocess.run(["git", "merge-base", "--is-ancestor", core_commit, "HEAD"], cwd=repo, capture_output=True).returncode == 0
    return {"coreTreeHash": core_tree_hash(files, cpg_dir), "packSetHash": pack_set_hash(files, cpg_dir), "harnessHash": harness_hash(),
            "head": _git("rev-parse", "HEAD", cwd=repo).strip(), "dirty": dirty, "clean": not dirty, "coreCommitIsAncestorOfHead": anc}


def tree_differences(rec: Mapping[str, Any], state: Mapping[str, Any]) -> list[str]:
    """The reasons a run is not the pre-registered one (empty: it is)."""
    out = []
    for key in ("coreTreeHash", "packSetHash", "harnessHash"):
        if state.get(key) != rec.get(key):
            out.append(f"{key} differs from the record")
    if not state.get("clean"):
        out.append("uncommitted changes under cpg/{ccg,clients,eval/tools,packs}")
    if state.get("coreCommitIsAncestorOfHead") is not True:
        out.append("the record's coreCommit is not an ancestor of HEAD")
    return out


# --------------------------------------------------------------------------------------------------------------- the record

ATTEMPT_KEYS = {"model", "counts", "unusable", "resultSha256"}
ATTEMPT_OPTIONAL_KEYS = {"promptHash"}


def _is_count(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def validate_dev_attempts(obj: Any) -> list[dict[str, Any]]:
    """``devAttempts`` (``docs/plans/connected-code-graph-p1-interpreter-model-ml.md``: every try, passed or failed): a list of ``{model, counts, unusable,
    resultSha256}`` plus an optional ``promptHash`` (64 lowercase hex when present), no other keys. ``model`` a non-empty string, ``counts`` a non-empty object of non-negative integers, ``unusable`` a non-negative integer,
    ``resultSha256`` 64 lowercase hex. Anything else is a :class:`Refusal`; the entries are returned as given."""
    if not isinstance(obj, list) or not obj:
        raise Refusal("dev attempts: the file holds a non-empty JSON list of attempts")
    for i, a in enumerate(obj):
        why = None
        if not isinstance(a, dict) or not (ATTEMPT_KEYS <= set(a) <= ATTEMPT_KEYS | ATTEMPT_OPTIONAL_KEYS):
            why = f"keys are {sorted(ATTEMPT_KEYS)} plus optionally {sorted(ATTEMPT_OPTIONAL_KEYS)}"
        elif not (isinstance(a["model"], str) and a["model"]):
            why = "model is a non-empty string"
        elif not (isinstance(a["counts"], dict) and a["counts"] and all(_is_count(v) for v in a["counts"].values())):
            why = "counts is a non-empty object of non-negative integers"
        elif not _is_count(a["unusable"]):
            why = "unusable is a non-negative integer"
        elif not (isinstance(a["resultSha256"], str) and re.fullmatch(r"[0-9a-f]{64}", a["resultSha256"])):
            why = "resultSha256 is 64 lowercase hex characters"
        elif "promptHash" in a and not (isinstance(a["promptHash"], str) and re.fullmatch(r"[0-9a-f]{64}", a["promptHash"])):
            why = "promptHash, when present, is 64 lowercase hex characters"
        if why:
            raise Refusal(f"dev attempts: attempt {i} is malformed ({why})")
    return obj


def build_prereg_record(dev_result: Mapping[str, Any], *, eval_commit: str, core_commit: str, core_tree_hash: str, pack_set_hash: str, harness_hash: str,
                        heldout_list_sha256: str, sampler_effective: Mapping[str, Any] | None = None, lm_studio_version: str | None = None,
                        recorded_by: str = "teco", dev_attempts: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """The ``ccg-prereg-ac24-v1`` record (EP §12.3; ``-p1-ml.md`` §4.3) from the final dev run's result. ``interpreter`` is the provenance that run observed;
    ``devExtras`` its ambiguous items' ``|options|`` and extras; ``canary`` its canary digest. **Returns the record; never appends it** (``teco`` appends)."""
    interp = dev_result.get("interpreterObserved")
    if not interp:
        raise Refusal("the dev run observed no provenance (every item was parser-resolved or none ran): there is no interpreter to record")
    if not dev_result.get("canary"):
        raise Refusal("the dev run has no canary digest")
    extras = dev_result["summary"]["ambiguousExtras"]
    rec = {
        "schema": PREREG_SCHEMA, "evalCommit": eval_commit, "coreCommit": core_commit, "coreTreeHash": core_tree_hash, "packSetHash": pack_set_hash,
        "harnessHash": harness_hash,
        "interpreter": {k: interp[k] for k in INTERPRETER_KEYS},
        "capFloor": CAP_FLOOR, "capRule": CAP_RULE,
        "devExtras": [{"qid": q, "options": e["nOptions"], "extras": e["extras"]} for q, e in sorted(extras.items())],
        "cacheBypass": CACHE_BYPASS_NAME, "tieReading": None,
        "sampler": {"request": dev_result["samplerRequest"], "effective": dict(sampler_effective) if sampler_effective is not None else None,
                    "effectiveStatus": "observed" if sampler_effective is not None else "not-observable"},
        "canary": dev_result["canary"],
        "repeatProtocol": REPEAT_PROTOCOL, "unusableObservable": UNUSABLE_OBSERVABLE,
        "heldoutListSha256": heldout_list_sha256, "expectedHeldoutCounts": dict(EXPECTED_HELDOUT),
        "lmStudioVersion": lm_studio_version, "lmStudioVersionStatus": "observed" if lm_studio_version else "not-obtainable",
        "directionCues": DIRECTION_CUES, "unbuiltReadingsRule": UNBUILT_READINGS_RULE, "resolverRules": list(RESOLVER_RULES),
        "recordedBy": recorded_by,
    }
    if dev_attempts is not None:
        rec["devAttempts"] = [dict(a) for a in validate_dev_attempts(list(dev_attempts))]
    return rec


# --------------------------------------------------------------------------------------------------------------- guards

def _git_head_text(path: Path) -> str:
    """``HEAD``'s copy of ``path`` (read-only ``git show``); raises :class:`Refusal` when it is not tracked there."""
    path = path.resolve()
    try:
        top = subprocess.run(["git", "-C", str(path.parent), "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True).stdout.strip()
        rel = path.relative_to(Path(top).resolve())
        return subprocess.run(["git", "-C", top, "show", f"HEAD:{rel.as_posix()}"], capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        raise Refusal("the pre-registration file is not committed (no HEAD copy)") from None


def find_prereg(prereg_path: Path, *, committed_text: Callable[[Path], str] = _git_head_text, eval_commit: str | None = None) -> dict[str, Any]:
    """The one ``ccg-prereg-ac24-v1`` record, present in the file AND in its committed (HEAD) copy; else :class:`Refusal`."""
    prereg_path = Path(prereg_path)
    if not prereg_path.exists():
        raise Refusal("no preregistration file: the ccg-prereg-ac24-v1 record is not present")
    recs = []
    for line in prereg_path.read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if isinstance(d, dict) and d.get("schema") == PREREG_SCHEMA:
            recs.append((line, d))
    if len(recs) != 1:
        raise Refusal(f"the preregistration file holds {len(recs)} {PREREG_SCHEMA} records (exactly one is required)")
    line, rec = recs[0]
    if line.strip() not in committed_text(prereg_path).splitlines():
        raise Refusal("the ccg-prereg-ac24-v1 record is present but not committed (HEAD's copy does not hold it)")
    interp = rec.get("interpreter")
    if not isinstance(interp, dict) or any(not interp.get(k) for k in INTERPRETER_KEYS):
        raise Refusal("the record's interpreter block is incomplete")
    if rec.get("capFloor") != CAP_FLOOR or rec.get("capRule") != CAP_RULE or rec.get("cacheBypass") != CACHE_BYPASS_NAME:
        raise Refusal("the record's capFloor, capRule or cacheBypass differs from the protocol's")
    if eval_commit is not None and rec.get("evalCommit") != eval_commit:
        raise Refusal("the record's evalCommit is not this repository's EVAL_COMMIT")
    for k in ("coreTreeHash", "packSetHash", "harnessHash", "heldoutListSha256", "coreCommit"):
        if not isinstance(rec.get(k), str) or not rec[k]:
            raise Refusal(f"the record has no {k}")
    if not isinstance(rec.get("canary"), dict) or not rec["canary"].get("digest"):
        raise Refusal("the record has no canary digest")
    if rec.get("expectedHeldoutCounts") != EXPECTED_HELDOUT:
        raise Refusal("the record's expectedHeldoutCounts is not this harness's")
    smp = rec.get("sampler")
    if not isinstance(smp, dict) or not isinstance(smp.get("request"), dict) or not isinstance(smp["request"].get("params"), dict) or "effective" not in smp:
        raise Refusal("the record's sampler block is incomplete")
    return rec


def custodian_area(home: Path | None = None) -> Path:
    return (home or Path.home()) / ".local" / "share" / "ccg-eval"


def custodian_root(eval_commit_file: Path | None = None, home: Path | None = None) -> Path:
    """``~/.local/share/ccg-eval/<E12>/`` (``-p1-ml.md`` §6.4); ``<E12>`` is the first 12 hex of ``EVAL_COMMIT``."""
    f = eval_commit_file or _HERE.parent / "EVAL_COMMIT"
    return custodian_area(home) / f.read_text().strip()[:12]


def refuse_custodian_path(path: Path, home: Path | None = None) -> None:
    """A dev run never reads from the custodian area (the held-out list lives there)."""
    area = custodian_area(home).resolve()
    p = Path(path).resolve()
    if p == area or area in p.parents:
        raise Refusal("the dev items path is under the custodian area: the dev run never reads held-out material")


def take_marker(marker: Path, payload: Mapping[str, Any]) -> None:
    """The once-only guard: create the marker atomically; an existing one (whatever it says) refuses the run."""
    marker.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise Refusal("the held-out AC-24 run was already started (marker present): a failing run is reported as run and the fix goes on dev items; "
                      "only a human removes the marker, after a harness error") from None
    with os.fdopen(fd, "w") as fh:
        fh.write(canonical_json(payload) + "\n")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path, what: str) -> dict[str, Any]:
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise Refusal(f"{what} is missing or not valid JSON") from None
    if not isinstance(d, dict):
        raise Refusal(f"{what} is not a JSON object")
    return d


# --------------------------------------------------------------------------------------------------------------- modes

class Rig:
    """What a run needs from the system under test: the client, the request log (with the draft events), the cache's length and (optionally) the effective-sampler observer."""

    def __init__(self, client: Any, log: RequestLog, cache_len: Callable[[], int], sampler_observer: Callable[[], Mapping[str, Any]] | None = None):
        self.client, self.log, self.cache_len, self.sampler_observer = client, log, cache_len, sampler_observer


def sampler_request() -> dict[str, Any]:
    """The request-body sampler (``ccg.front.interpret.SAMPLING``) and its ``samplingHash``."""
    from ccg.front import interpret
    return {"params": dict(interpret.SAMPLING), "samplingHash": interpret.sampling_hash()}


def run_dev(dev_items: Path, rig: Rig, out: Path, *, prereg_interpreter: Mapping[str, Any] | None = None, home: Path | None = None,
            tree_state: Callable[[str | None], Mapping[str, Any]] = observe_tree) -> dict[str, Any]:
    """The dev entry run. The result carries ``treeStamp``: :func:`observe_tree` taken before the run and re-taken after it (a difference is a harness error), the
    only source ``prereg-record`` takes its three hashes from (N1)."""
    refuse_custodian_path(dev_items, home)
    items = load_items(dev_items)
    counts = recount(items)
    if counts["ambiguous"] != 2 or counts["unambiguous"] != 12:
        raise Refusal(f"the dev set holds {counts['unambiguous']} unambiguous and {counts['ambiguous']} ambiguous items, not 12 and 2")
    stamp = dict(tree_state(None))
    canary = run_canary(rig.client, rig.log)
    run = run_items(items, rig.client, repeats=1, log=rig.log, cache_len=rig.cache_len, expected_interpreter=prereg_interpreter,
                    sampler_observer=rig.sampler_observer, record_drafts=True)
    if dict(tree_state(None)) != stamp:
        raise HarnessError("the tree (hashes, HEAD or cleanliness) changed during the dev run: no stamp can describe it")
    s = summarize(run, items, mode="dev")
    result = {**{k: v for k, v in run.items() if k != "items"}, "items": run["items"], "summary": s, "samplerRequest": sampler_request(), "canary": canary,
              "treeStamp": stamp}
    _write_json(out, result)
    return result


def run_heldout(*, explicit: bool, root: Path, marker_dir: Path, prereg_path: Path, dev_result_path: Path, rig_factory: Callable[[], Rig],
                committed_text: Callable[[Path], str] = _git_head_text, eval_commit: str, tree_state: Callable[[str | None], Mapping[str, Any]] = observe_tree,
                harness_sha: Callable[[], str] = harness_hash, manifest_path: Path | None = None, after_adaptation: bool = False) -> dict[str, Any]:
    """The held-out run. Every check that does not need the items, and the rig, come before the once-only marker; the marker precedes the items. ``marker_dir`` is the
    DERIVED custodian root and ``eval_commit`` the repository's: the marker's name carries the E12, so copying ``heldout/`` elsewhere gives no fresh marker."""
    if explicit is not True:
        raise Refusal("the held-out run needs the explicit --heldout flag")
    rec = find_prereg(prereg_path, committed_text=committed_text, eval_commit=eval_commit)
    dev = _load_json(dev_result_path, "the dev result")
    if dev.get("summary", {}).get("entryPass") is not True:
        raise Refusal("the dev entry check has no passing result: no held-out run")
    if dev.get("interpreterObserved") != rec["interpreter"]:
        raise Refusal("the dev run's interpreter is not the one the record pre-registers")
    if dev.get("canary") != rec["canary"]:
        raise Refusal("the dev run's canary is not the one the record pre-registers")
    items_path = Path(root) / "heldout" / "ac24.jsonl"
    if not items_path.exists():
        raise Refusal("no held-out AC-24 list under the custodian root")
    list_sha = _file_sha(items_path)                                    # the bytes' hash, never the text
    if list_sha != rec["heldoutListSha256"]:
        raise Refusal("the held-out list's sha256 is not the record's")
    if manifest_path is not None:
        listed = [f.get("sha256") for f in _load_json(manifest_path, "the held-out manifest").get("files", []) if f.get("name") == "ac24.jsonl"]
        if listed != [list_sha]:
            raise Refusal("the held-out list's sha256 is not the one heldout-manifest.json lists")
    state = dict(tree_state(rec["coreCommit"]))
    reasons = tree_differences(rec, {**state, "harnessHash": harness_sha()})
    if reasons and not after_adaptation:
        raise Refusal("the run is not the pre-registered one (" + "; ".join(reasons) + "); --after-adaptation stamps it instead")
    rig = rig_factory()
    smp = rec["sampler"]
    if smp["effective"] is not None and rig.sampler_observer is None:
        raise Refusal("the record pre-registers an effective sampler configuration and this rig has no observer to read it")
    if rig.cache_len() != 0:
        raise HarnessError("the interpretation cache is not empty before repeat 1 (a fresh cache per gate run)")
    effective = check_sampler_effective(rig.sampler_observer, smp["effective"])
    canary = run_canary(rig.client, rig.log)
    if canary != rec["canary"]:
        raise HarnessError("the canary digest differs from the pre-registered one (the model, template or sampler changed since the dev run)")
    marker = Path(marker_dir) / f"ac24-heldout-once-{eval_commit[:12]}.json"
    take_marker(marker, {"state": "started", "evalCommit": eval_commit, "record": sha256_text(canonical_json(rec))})
    try:
        items = load_items(items_path)
        counts = recount(items)
        if {k: counts.get(k) for k in EXPECTED_HELDOUT} != EXPECTED_HELDOUT:
            raise Refusal(f"the held-out split holds {counts}, not {EXPECTED_HELDOUT}: the method owner recomputes the thresholds before the run")
        run = run_items(items, rig.client, repeats=REPEATS_HELDOUT, log=rig.log, cache_len=rig.cache_len, expected_interpreter=rec["interpreter"],
                        expected_sampler_request=smp["request"]["params"], sampler_observer=rig.sampler_observer, expected_sampler_effective=smp["effective"],
                        cap_floor=rec["capFloor"])
    except (HarnessError, Refusal) as exc:
        marker.write_text(canonical_json({"state": "aborted", "why": type(exc).__name__, "detail": str(exc)[:200]}) + "\n")
        raise
    s = summarize(run, items, mode="heldout")
    s["tree"] = {"head": state.get("head"), "clean": state.get("clean"), "afterAdaptation": bool(reasons), "afterAdaptationReasons": reasons,
                 "harnessHash": harness_sha(), "heldoutListSha256": list_sha, "canaryDigest": canary["digest"], "samplerEffective": effective}
    if reasons:
        s["verdict"] += " (after adaptation)"
    result = {**{k: v for k, v in run.items() if k != "items"}, "items": run["items"], "summary": s}
    _write_json(Path(root) / "ac24-heldout-run.json", result)
    marker.write_text(canonical_json({"state": "completed", "verdict": s["verdict"]}) + "\n")
    return result


# --------------------------------------------------------------------------------------------------------------- live wiring (not exercised by the tests)

def build_live_rig(system: str) -> Rig:
    """The real ``Client`` over the real ``Front`` and a recording ``Interpreter`` through a counting proxy of the broker client. The effective-sampler observer is not
    wired (no documented read of LM Studio's effective ``repeat_penalty``/``top_k``/``min_p``/preset): the record carries ``effective: null`` with status
    ``not-observable`` and the canary digest stands in for the read."""
    from ccg.broker.client import BrokerClient
    from ccg.front import Front, interpret
    from clients import common

    repos = common.system_repos(system)
    log = RequestLog()
    broker = CountingBroker(BrokerClient(), log)
    interp = recording_interpreter(log)
    holder: dict[str, Any] = {}
    front = Front(repos, lambda anchor, direction: holder["c"]._builder(holder["c"]._reader(), anchor, question=direction), interpreter=interp,
                  run_factory=interpret.run_factory(system, repos, broker))
    holder["c"] = common.Client(system, front=front)
    return Rig(holder["c"], log, lambda: len(interp.cache))


# --------------------------------------------------------------------------------------------------------------- CLI

def _print_summary(s: Mapping[str, Any], out: Path) -> None:
    a, b = s["unambiguousUnderstood"], s["ambiguousUnderstood"]
    pc = s["pooledComposition"]
    print(f"{s['verdict']}: unambiguous {a['k']}/{a['n']} {a['wilson95']}, ambiguous {b['k']}/{b['n']}; over-asking {s['overAsking']['k']}/{s['overAsking']['n']}; "
          f"unusable {s['unusableObserved']['count']}; parser-resolved {s['parserResolved']['count']} (pooled: {pc['parserResolvedUnambiguous']} parser, "
          f"{pc['interpreterDrawsUnambiguous']} interpreter)")
    print(f"output: {out}")
    print(f"note: {CLAIM}")


def main(argv: Sequence[str] | None = None, *, rig_factory: Callable[[str], Rig] = build_live_rig, committed_text: Callable[[Path], str] = _git_head_text,
         tree_state: Callable[[str | None], Mapping[str, Any]] = observe_tree, home: Path | None = None, eval_commit: str | None = None,
         manifest_path: Path | None = None, stdout=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="verb", required=True)
    d = sub.add_parser("dev")
    d.add_argument("--dev-items", type=Path, default=_CPG / "eval" / "dev" / "ac24.jsonl")
    d.add_argument("--system", default="graphmind-ai-lab")
    d.add_argument("--out", type=Path, default=_CPG / "eval" / "results" / "ac24-dev-run.json")
    d.add_argument("--prereg", type=Path)
    c = sub.add_parser("canary")
    c.add_argument("--system", default="graphmind-ai-lab")
    sub.add_parser("tree-hashes")
    h = sub.add_parser("heldout")
    h.add_argument("--heldout", action="store_true", dest="explicit")
    h.add_argument("--custodian-root", type=Path)
    h.add_argument("--prereg", type=Path, default=_CPG / "eval" / "preregistration.jsonl")
    h.add_argument("--dev-result", type=Path, required=True)
    h.add_argument("--system", default="graphmind-ai-lab")
    h.add_argument("--after-adaptation", action="store_true")
    p = sub.add_parser("prereg-record")
    p.add_argument("--dev-result", type=Path, required=True)
    for k in ("eval-commit", "core-commit"):
        p.add_argument(f"--{k}", required=True)
    p.add_argument("--heldout-list-sha256")
    p.add_argument("--out", type=Path)
    p.add_argument("--dev-attempts", type=Path, help="JSON file: a list of {model, counts, unusable, resultSha256}, copied into the record as devAttempts")
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 2 if e.code else 0
    out = stdout or sys.stdout
    try:
        if a.verb == "dev":
            pre = None
            if a.prereg:
                pre = find_prereg(a.prereg, committed_text=committed_text)["interpreter"]
            res = run_dev(a.dev_items, rig_factory(a.system), a.out, prereg_interpreter=pre, home=home, tree_state=tree_state)
            _print_summary(res["summary"], a.out)
            return 0 if res["summary"]["entryPass"] else 1
        if a.verb == "canary":
            rig = rig_factory(a.system)
            print(canonical_json(run_canary(rig.client, rig.log)), file=out)
            return 0
        if a.verb == "tree-hashes":
            print(canonical_json({k: v for k, v in tree_state(None).items()}), file=out)
            return 0
        if a.verb == "heldout":
            derived = custodian_root(home=home)
            if a.custodian_root is not None and a.custodian_root.resolve() != derived.resolve():
                raise Refusal("--custodian-root is not the root derived from EVAL_COMMIT: the marker lives only in the derived root")
            ev = eval_commit or (_HERE.parent / "EVAL_COMMIT").read_text().strip()
            res = run_heldout(explicit=a.explicit, root=derived, marker_dir=derived, prereg_path=a.prereg, dev_result_path=a.dev_result,
                              rig_factory=lambda: rig_factory(a.system), committed_text=committed_text, eval_commit=ev, tree_state=tree_state,
                              manifest_path=manifest_path or _HERE.parent / "heldout-manifest.json", after_adaptation=a.after_adaptation)
            _print_summary(res["summary"], derived / "ac24-heldout-run.json")
            return 0 if res["summary"]["passed"] else 1
        if a.out and (a.out.exists() or a.out.name == "preregistration.jsonl"):
            raise Refusal("--out exists or names preregistration.jsonl: the record is never written over a file (teco appends it)")
        dev = _load_json(a.dev_result, "the dev result")
        attempts = None
        if a.dev_attempts is not None:
            try:
                attempts = json.loads(a.dev_attempts.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise Refusal(f"dev attempts: {a.dev_attempts} is unreadable or not JSON ({type(exc).__name__})") from None
        stamp = dev.get("treeStamp")
        if not isinstance(stamp, dict) or not all(isinstance(stamp.get(k), str) for k in ("coreTreeHash", "packSetHash", "harnessHash")):
            raise Refusal("the dev result carries no tree stamp: run the dev verb again from a clean committed tree")
        st = tree_state(None)
        problems = [f"{k} differs from the dev run's stamp" for k in ("coreTreeHash", "packSetHash", "harnessHash") if st.get(k) != stamp[k]]
        if stamp.get("clean") is not True:
            problems.append("the dev run's tree was not clean")
        if not st.get("clean"):
            problems.append("the tree is not clean")
        if problems:
            raise Refusal("the record is not built from the tree the dev run ran on: " + "; ".join(problems))
        list_sha = a.heldout_list_sha256 or _file_sha(custodian_root(home=home) / "heldout" / "ac24.jsonl")
        rec = build_prereg_record(dev, eval_commit=a.eval_commit, core_commit=a.core_commit, core_tree_hash=stamp["coreTreeHash"],
                                  pack_set_hash=stamp["packSetHash"], harness_hash=stamp["harnessHash"], heldout_list_sha256=list_sha, dev_attempts=attempts)
        line = canonical_json(rec)
        if a.out:
            a.out.write_text(line + "\n", encoding="utf-8")
            print(f"record written to {a.out} (not appended to preregistration.jsonl)", file=out)
        else:
            print(line, file=out)
        return 0
    except Refusal as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 4
    except HarnessError as exc:
        print(f"harness error (no score): {exc}", file=sys.stderr)
        return 5


if __name__ == "__main__":
    sys.exit(main())
