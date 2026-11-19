"""The interpreter of the plain-language front (C-1001 P1-26 behaviours (2), (4), (6); ``docs/plans/connected-code-graph-p1-ml.md`` §4.2, ``-egress-spec3.md`` §2.1).

For free prose (what the deterministic parsers did not read) one model call, through the egress broker and nothing else, extracts an ``InterpretDraft`` (spans, never ids).
The reply is grammar-constrained by LM Studio's ``response_format`` ``json_schema`` (``grammar/interpret.json``; for GGUF models LM Studio's documentation says llama.cpp's
grammar-based sampling enforces it, and it warns that models under 7B are not all capable of it) and validated against that same schema here. A reply that is not valid
JSON, or does not validate, is **not repaired by a second call**: the question is read as naming nothing (:class:`Interpretation` with ``draft=None``). The fabrication
guard (every span a verbatim substring of the question) lives in ``ccg.front.resolve`` and runs on every draft, cached or fresh.

* **Run.** :class:`ModelRun` is one question's broker ``answer`` run: opened at the first model call (a fingerprint read, a completion, a summary) and reused for every later one,
  never opened for a question the parsers resolve; :meth:`ModelRun.close` closes it. A broker that is absent, latched or refusing fails the call with :class:`ModelError`;
  there is no direct connection in this module (``ccg.broker.client`` is the only broker import).
* **Sampling.** :data:`SAMPLING` is the one table both the request body and ``samplingHash`` are built from. Only a transport failure (no completed exchange, or a 5xx) is
  retried, once, and the retry is logged.
* **Provenance.** The fingerprint ``id|publisher|arch|quantization|loaded_context_length`` is read from ``/api/v0/models`` at call time (the bound model is whatever the endpoint
  reports; :data:`RECOMMENDED_MODEL` is preferred when loaded); the prompt and grammar hashes are sha256 of the files; ``cacheKey`` is :func:`ccg.front.cache.key_of`.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from ccg import answer
from ccg.broker.client import BrokerClient, BrokerError, BrokerUnavailable, RunGrant
from ccg.front import cache as cache_mod
from ccg.front.draft import Draft, Group, Reading, Span
from ccg.front.errors import ModelError
from ccg.front.resolve import RESOLVER_VERSION

HERE = Path(__file__).resolve().parent
LOCAL_ENDPOINT = "http://127.0.0.1:1234"
MODELS_PATH = "/api/v0/models"
CHAT_PATH = "/v1/chat/completions"
RECOMMENDED_MODEL = "qwen/qwen3-4b-2507"
COMPONENT = "answer.interpret"
PURPOSE = "model-call"
#: the sampler of the interpreter (``-p1-ml.md`` §4.2): the request body and ``samplingHash`` both come from this table. Every parameter LM Studio's chat-completions
#: documentation lists that shapes decoding is sent explicitly at its neutral value (``top_k`` 0 and ``repeat_penalty`` 1.0 are the llama.cpp "off" values), so a model
#: preset's own default is never in effect. ``min_p`` is not in that documentation's list, so it is not sent.
SAMPLING: dict[str, int | float] = {"temperature": 0, "top_p": 1, "top_k": 0, "repeat_penalty": 1.0, "frequency_penalty": 0, "presence_penalty": 0,
                                    "max_tokens": 384, "seed": 0}
#: the fractional parameters, hashed in milli-units as ``<name>_milli`` (the temperature is hashed first-class)
MILLI_PARAMS = frozenset({"top_p", "repeat_penalty", "frequency_penalty", "presence_penalty"})
RETRIES = 1
LOG = logging.getLogger("ccg.front.model")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prompt_text() -> str:
    return (HERE / "prompts" / "interpret.txt").read_text(encoding="utf-8")


def grammar() -> dict[str, Any]:
    return json.loads((HERE / "grammar" / "interpret.json").read_text(encoding="utf-8"))


def prompt_hash() -> str:
    return _sha256(HERE / "prompts" / "interpret.txt")


def grammar_hash() -> str:
    return _sha256(HERE / "grammar" / "interpret.json")


def sampling_hash(sampling: dict[str, int | float] | None = None) -> str:
    """``ccg.answer.sampling_hash`` over the sampler: the temperature in milli-units, every other parameter as an integer (the :data:`MILLI_PARAMS` in milli-units as ``<name>_milli``)."""
    s = dict(SAMPLING if sampling is None else sampling)
    temperature = s.pop("temperature")
    params = {(f"{k}_milli" if k in MILLI_PARAMS else k): round(v * 1000) if k in MILLI_PARAMS else int(v) for k, v in s.items()}
    return answer.sampling_hash(round(temperature * 1000), params)


def fingerprint(model: dict[str, Any]) -> str:
    return "|".join(str(model.get(k, "")) for k in ("id", "publisher", "arch", "quantization", "loaded_context_length"))


def pick_model(models: list[Any]) -> dict[str, Any]:
    """The loaded chat model the endpoint reports (the recommended one when loaded, else the first by id)."""
    loaded = sorted((m for m in models if isinstance(m, dict) and m.get("state") == "loaded" and m.get("type") in ("llm", "vlm") and isinstance(m.get("id"), str)),
                    key=lambda m: m["id"])
    if not loaded:
        raise ModelError("the model endpoint reports no loaded chat model", detail={"endpoint": "local"})
    return next((m for m in loaded if m["id"] == RECOMMENDED_MODEL), loaded[0])


class ModelRun:
    """One question's broker run and its model calls (opened lazily, reused, closed by :meth:`close`). ``broker`` is a ``BrokerClient`` (or a double of its three methods)."""

    def __init__(self, broker: BrokerClient, system: str | None, repos: tuple[str, ...], endpoint: str = LOCAL_ENDPOINT):
        self._broker, self._system, self._repos, self._endpoint = broker, system, tuple(repos), endpoint
        self._grant: RunGrant | None = None
        self._model: dict[str, Any] | None = None
        self.opened = 0

    def _token(self) -> str:
        if self._grant is None:
            if not self._system or not self._repos:
                raise ModelError("no system with repositories is known, so no model run can be opened")
            try:
                self._grant = self._broker.open_run("answer", self._system, "*")
            except BrokerUnavailable as exc:
                raise ModelError("the egress broker is not reachable, so no model call is made", detail={"cause": type(exc).__name__}) from None
            except BrokerError as exc:
                raise ModelError(f"the egress broker refused the run ({exc.code})", detail={"cause": exc.code}) from None
            self.opened += 1
        return self._grant.token

    def _send(self, component: str, method: str, path: str, body: Any) -> Any:
        token = self._token()
        req = {"component": component, "purpose": PURPOSE, "payloadClass": "none" if body is None else "repo-derived", "endpoint": self._endpoint,
               "target": {"method": method, "path": path}, "codebases": list(self._repos)}
        for attempt in range(RETRIES + 1):
            try:
                decision, resp = self._broker.request(token, req, body)
            except BrokerUnavailable as exc:
                raise ModelError("the egress broker is not reachable, so no model call is made", detail={"cause": type(exc).__name__}) from None
            except BrokerError as exc:
                raise ModelError(f"the egress broker refused the call ({exc.code})", detail={"cause": exc.code}) from None
            if not decision.allowed:
                raise ModelError(decision.refusal(), detail={"rule": decision.rule, "reason": decision.reason})
            if resp is not None and resp.status < 500:
                break
            LOG.warning("model call transport failure (%s %s, attempt %d of %d)", method, path, attempt + 1, RETRIES + 1)
        else:
            raise ModelError("the model endpoint did not answer", detail={"status": getattr(resp, "status", None)})
        if resp.status != 200:
            raise ModelError(f"the model endpoint answered HTTP {resp.status}", detail={"status": resp.status})
        try:
            return json.loads(resp.body, parse_constant=_no_constant)
        except ValueError:
            raise ModelError("the model endpoint sent a reply that is not JSON") from None

    def bound_model(self, component: str = COMPONENT) -> dict[str, Any]:
        """The loaded model the endpoint reports, read once per run (a ``GET`` through it, declared by whichever component reads first) and shared by the interpreter and the summary."""
        if self._model is not None:
            return self._model
        reply = self._send(component, "GET", MODELS_PATH, None)
        models = reply.get("data") if isinstance(reply, dict) else None
        if not isinstance(models, list):
            raise ModelError("the model endpoint's model list is malformed")
        self._model = pick_model(models)
        return self._model

    def complete(self, component: str, body: dict[str, Any]) -> str:
        """One chat completion; the reply's message content."""
        reply = self._send(component, "POST", CHAT_PATH, body)
        try:
            content = reply["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise ModelError("the model endpoint's reply has no message") from None
        if not isinstance(content, str):
            raise ModelError("the model endpoint's reply has no text")
        return content

    def close(self) -> None:
        grant, self._grant = self._grant, None
        if grant is not None:
            try:
                self._broker.close_run(grant.token)
            except Exception:  # noqa: BLE001 - closing is best effort; the run also expires
                LOG.warning("closing the model run failed")


def run_factory(system: str | None, repos: tuple[str, ...], broker: BrokerClient | None = None):
    """``() -> ModelRun``: a fresh run per question over the broker client (the default socket unless ``broker`` is given); no connection is made until the first call."""
    return lambda: ModelRun(broker or BrokerClient(), system, tuple(repos))


def _no_constant(name: str):
    raise ValueError(name)


@dataclass(frozen=True)
class Interpretation:
    """``draft`` is None when the reply was unusable (not JSON, or not valid against the grammar); ``provenance`` is the answer's ``interpretation.provenance``."""
    draft: Draft | None
    provenance: dict[str, Any]


def to_draft(reply: dict[str, Any]) -> Draft | None:
    try:
        return Draft(reply["useCase"], reply["direction"],
                     tuple(Group(tuple(Span(s["text"], s["kindHint"]) for s in g["spans"])) for g in reply["anchors"]),
                     scope=reply["scope"], readings=tuple(Reading(r["useCase"], r["direction"]) for r in reply["readings"]))
    except (KeyError, TypeError, ValueError):
        return None


class Interpreter:
    """Free prose to an :class:`Interpretation`: the cache, then one model call. ``cache`` is shared by the questions of one process."""

    def __init__(self, cache: cache_mod.InterpretCache | None = None):
        self.cache = cache if cache is not None else cache_mod.InterpretCache()
        self._validator = Draft202012Validator(grammar())

    def request_body(self, model_id: str, question: str) -> dict[str, Any]:
        return {"model": model_id, "messages": [{"role": "system", "content": prompt_text()}, {"role": "user", "content": question}],
                **SAMPLING, "stream": False,
                "response_format": {"type": "json_schema", "json_schema": {"name": "interpret_draft", "strict": True, "schema": grammar()}}}

    def _valid_reply(self, content: str) -> dict[str, Any] | None:
        try:
            reply = json.loads(content, parse_constant=_no_constant)
        except ValueError:
            return None
        return reply if isinstance(reply, dict) and not any(self._validator.iter_errors(reply)) else None

    def interpret(self, question: str, snapshot_id: str, run: ModelRun, *, bypass: bool = False) -> Interpretation:
        """``question`` is the normalized question. With ``bypass`` the cache is neither read nor written, the interpreter is called, and the provenance records it."""
        model = run.bound_model()
        fp = fingerprint(model)
        key = cache_mod.key_of(question, snapshot_id, fp, prompt_hash(), grammar_hash(), sampling_hash(), RESOLVER_VERSION)
        reply = None if bypass else self.cache.get(key, snapshot_id)
        hit = reply is not None
        if reply is None:
            reply = self._valid_reply(run.complete(COMPONENT, self.request_body(model["id"], question)))
            if reply is not None and not bypass:
                self.cache.put(key, snapshot_id, reply)
        prov: dict[str, Any] = {"modelFingerprint": fp, "promptHash": prompt_hash(), "grammarHash": grammar_hash(), "samplingHash": sampling_hash(),
                                "resolverVersion": RESOLVER_VERSION, "endpoint": "local", "temperatureMilli": round(SAMPLING["temperature"] * 1000),
                                "quantization": str(model.get("quantization") or "unknown"), "cacheKey": key, "cacheHit": hit}
        if bypass:
            prov["cacheBypass"] = True
        return Interpretation(to_draft(reply) if reply is not None else None, prov)
