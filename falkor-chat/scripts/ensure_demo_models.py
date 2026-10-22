#!/usr/bin/env python3
"""Ensure LM Studio has the storefront demo's models loaded before serving (K-066).

Run through `scripts/ensure_demo_models.sh` (which uses `server/.venv/bin/python`);
`scripts/start_demo.sh` calls it before launching uvicorn.

What it does, in order — every refusal happens before any load:

  1. Derives the demo's model set from the code that owns it, never a copy:
     the chat model is each `agent` step's `config.model` in
     `falkorchat.proof_defs.SALESPERSON_DEF` (the same model `query_graph_data`'s
     internal call inherits), and the embedder is the overlay's `embedding` default
     (`FALKORCHAT_MODEL_CONFIG`). Both are resolved through `ModelGateway` with the same
     `FALKORCHAT_OPENCODE_CONFIG` the server gets, so the LM Studio URL checked is the
     one the server will call.
     LIMIT: workspace model overrides on `ws:demo` are NOT read. An override is a hard
     cap over `config.model`, so an `agentModelOverride` set there would make the
     storefront call a model this step never checks. No script sets one; the only
     override `ws:demo` carries is the embedding pin `create_workspace.sh` writes, which
     equalled the overlay default at pin time (it is never re-pinned).
  2. Reads what is loaded: `GET <host>/api/v0/models` (portable, read-only). Unreachable,
     or any HTTP error other than 404/405 (a sick LM Studio) -> stop (exit 1). A server
     that answers 404/405, a non-JSON body, or JSON without a `data` list is not LM Studio
     -> stop with exit 3, the one case where skipping this step (`DEMO_ENSURE_MODELS=0`)
     is legitimate.
  3. Refuses (exit 1, nothing loaded or unloaded) when:
       * any model OUTSIDE the demo set is loaded (e.g. `qwen/qwen3-4b-2507`, which the
         `:8200` agent-team server may be using) — the operator decides what to unload;
       * the chat model is loaded with a context length other than DEMO_CONTEXT_LENGTH.
         The embedder's context is NOT checked: the API misreports it for embedding
         models (2048 for an instance `lms ps --json` shows at 8192, the value the
         runtime log in `docs/test-reports/salesperson-single-model-fit-report.md` also
         shows). Cost of the skip: an embedder at a wrong context passes, which the
         storefront cannot notice — it never embeds. GPU offload is not reported by the
         API or `lms ps`, so it is set at load only;
       * a demo model is still `loading` (wait and re-run).
  4. Nothing missing -> prints "nothing to do", exit 0 (idempotent; no CLI needed).
  5. Otherwise loads each missing model with the `lms` CLI —
     `lms load <id> -c 8192 --gpu max --identifier <id> -y` — then re-reads the API
     (while a demo model still reports `loading`, every SETTLE_INTERVAL_S for up to
     SETTLE_S = 30 s) and stops unless the full set is loaded and nothing foreign appeared. `--identifier`
     makes a racing duplicate load fail instead of silently creating `<id>:2`.

It NEVER runs `lms unload`, never reloads a loaded model, and never JIT-loads by
calling a model.

The CLI is found by configuration: `LMS_CLI` (path to `lms`/`lms.exe`) if set, else
`lms` then `lms.exe` on PATH. It is only needed when something must be loaded; when it
is absent the step stops with load-it-yourself instructions.

Exit codes: 0 = demo set loaded with the demo settings; 1 = refused or failed (message
says why and what to do); 2 = usage; 3 = the URL answered but is not LM Studio.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

_REPO_ROOT = Path(__file__).resolve().parents[1]  # .../falkor-chat
_SERVER_DIR = _REPO_ROOT / "server"
if str(_SERVER_DIR) not in sys.path:  # direct run outside the venv's own resolution
    sys.path.insert(0, str(_SERVER_DIR))

# The demo settings — the configuration measured in
# docs/test-reports/salesperson-single-model-fit-report.md (8192 is also the unified
# KV pool concurrent storefront turns share).
DEMO_CONTEXT_LENGTH = 8192
DEMO_GPU = "max"

API_TIMEOUT_S = 5.0
LOAD_TIMEOUT_S = 300.0
# Post-load wait while a demo model still reports `loading` (env-overridable for tests).
SETTLE_S = float(os.environ.get("ENSURE_DEMO_MODELS_SETTLE_S", "30"))
SETTLE_INTERVAL_S = float(os.environ.get("ENSURE_DEMO_MODELS_SETTLE_INTERVAL_S", "1"))

PREFIX = "ensure_demo_models:"

# Only "no such route" means another provider; any other HTTP error is a sick LM Studio,
# which must not be offered the bypass.
NOT_LMSTUDIO_HTTP_CODES = frozenset({404, 405})
_BYPASS_HINT = ("(the ensure step needs LM Studio's /api/v0; set DEMO_ENSURE_MODELS=0 in "
                "start_demo.sh to skip it for another provider)")


class Refusal(Exception):
    """A stop with an operator-facing message; nothing was unloaded."""

    exit_code = 1


class NotLMStudio(Refusal):
    """The URL answered, but without LM Studio's `/api/v0` — the only case where
    skipping the step (`DEMO_ENSURE_MODELS=0`) is the right advice."""

    exit_code = 3


@dataclass(frozen=True)
class DemoSet:
    api_root: str            # e.g. http://localhost:1234 (no /v1)
    chat: tuple[str, ...]    # LM Studio model ids whose context is enforced
    embedding: str

    @property
    def all_ids(self) -> tuple[str, ...]:
        return (*self.chat, self.embedding)


def _say(msg: str) -> None:
    print(f"{PREFIX} {msg}", flush=True)


# ── 1. derive the demo set from its owners ─────────────────────────────────────────

def derive_demo_set() -> DemoSet:
    try:
        from falkorchat.modelconfig import ModelGateway
        from falkorchat.proof_defs import SALESPERSON_DEF
    except Exception as exc:  # noqa: BLE001 — surfaced verbatim to the operator
        raise Refusal(f"cannot import falkorchat to derive the demo models: {exc!r}")

    try:
        gateway = ModelGateway.from_env()
        resolutions = [
            gateway.resolve("step", requested=(step.get("config") or {}).get("model"))
            for step in SALESPERSON_DEF["steps"]
            if step.get("type") == "agent"
        ]
        embedding = gateway.resolve("embedding")
    except Exception as exc:  # noqa: BLE001
        raise Refusal(
            f"cannot resolve the demo models through ModelGateway: {exc}\n"
            "  Check FALKORCHAT_OPENCODE_CONFIG / FALKORCHAT_MODEL_CONFIG."
        )
    if not resolutions:
        raise Refusal("SALESPERSON_DEF has no type 'agent' step — nothing to derive a chat model from")

    chat: list[str] = []
    roots: set[str] = set()
    for res in (*resolutions, embedding):
        if len(res.chain) != 1:
            raise Refusal(
                f"kind {res.kind!r} resolves to a {len(res.chain)}-model fallback chain "
                f"({', '.join(r.ref for r in res.chain)}); this step only ensures single models"
            )
        primary = res.primary
        parsed = urlparse(primary.base_url)
        path = parsed.path.rstrip("/")
        if path.endswith("/v1"):
            path = path[: -len("/v1")]
        roots.add(f"{parsed.scheme}://{parsed.netloc}{path}")
        if res is not embedding and primary.model not in chat:
            chat.append(primary.model)
    if len(roots) != 1:
        raise Refusal(f"demo models resolve to more than one server: {sorted(roots)}")
    return DemoSet(api_root=roots.pop(), chat=tuple(chat), embedding=embedding.primary.model)


# ── 2. read what is loaded ─────────────────────────────────────────────────────────

def read_models(api_root: str) -> list[dict]:
    url = f"{api_root}/api/v0/models"
    try:
        with urllib.request.urlopen(url, timeout=API_TIMEOUT_S) as resp:
            try:
                body = json.load(resp)
            except ValueError:  # reachable, but not JSON (e.g. an HTML catch-all host)
                raise NotLMStudio(f"{url} answered with a body that is not JSON — "
                                  f"is this an LM Studio server? {_BYPASS_HINT}")
    except urllib.error.HTTPError as exc:
        if exc.code in NOT_LMSTUDIO_HTTP_CODES:  # no such route: another provider
            raise NotLMStudio(f"{url} answered HTTP {exc.code} — is this an LM Studio "
                              f"server? {_BYPASS_HINT}")
        raise Refusal(f"LM Studio answered HTTP {exc.code} at {url} — check its server log "
                      "(Developer tab), then re-run.")
    except (urllib.error.URLError, OSError) as exc:
        raise Refusal(
            f"LM Studio API unreachable at {url}: {exc}\n"
            "  The URL is the lmstudio baseURL in FALKORCHAT_OPENCODE_CONFIG"
            f" ({os.environ.get('FALKORCHAT_OPENCODE_CONFIG', '<unset>')}).\n"
            "  Fix: point it at a reachable LM Studio — e.g. on a box whose shared opencode.json\n"
            "  is stale, export FALKORCHAT_OPENCODE_CONFIG=$HOME/.config/opencode/opencode.local.json\n"
            "  — or start LM Studio's server (Developer tab, or `lms server start`)."
        )
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list):
        raise NotLMStudio(f"{url} returned an unexpected body (no 'data' list) — is this an "
                          "LM Studio server?")
    bad = [m for m in data if not (isinstance(m, dict) and isinstance(m.get("id"), str))]
    if bad:
        raise Refusal(f"{url} returned {len(bad)} malformed model entr(y/ies), e.g. {bad[0]!r}")
    return data


def loaded_models(models: list[dict]) -> dict[str, dict]:
    """Every instance that is not `not-loaded` (so `loading` counts as present)."""
    return {m["id"]: m for m in models if m.get("state") != "not-loaded"}


def check_refusals(demo: DemoSet, loaded: dict[str, dict]) -> None:
    problems: list[str] = []
    foreign = sorted(set(loaded) - set(demo.all_ids))
    if foreign:
        problems.append(
            "model(s) outside the demo set are loaded: " + ", ".join(foreign) + "\n"
            "    They would compete for the GPU with the demo's models. This step never unloads;\n"
            "    if nothing needs them (the :8200 agent-team server may be using qwen), unload\n"
            "    them yourself, e.g.:  "
            + "  ".join(f"lms unload {fid}" for fid in foreign)
        )
    for mid in demo.all_ids:
        if loaded.get(mid, {}).get("state") == "loading":
            problems.append(f"{mid} is still loading — wait for it to finish and re-run.")
    for cid in demo.chat:
        inst = loaded.get(cid)
        if inst is None or inst.get("state") == "loading":
            continue
        ctx = inst.get("loaded_context_length")
        if ctx != DEMO_CONTEXT_LENGTH:
            problems.append(
                f"{cid} is loaded with context {ctx}, not the demo's {DEMO_CONTEXT_LENGTH}.\n"
                "    This step never reloads; unload it yourself and re-run:\n"
                f"      lms unload {cid}"
            )
    if problems:
        raise Refusal("refusing to continue:\n  - " + "\n  - ".join(problems))


# ── 5. load what is missing ────────────────────────────────────────────────────────

def load_args(model_id: str) -> list[str]:
    return [
        "load", model_id,
        "-c", str(DEMO_CONTEXT_LENGTH),
        "--gpu", DEMO_GPU,
        "--identifier", model_id,
        "-y",
    ]


def find_cli() -> str | None:
    configured = os.environ.get("LMS_CLI")
    if configured:
        if os.path.isfile(configured) and os.access(configured, os.X_OK):
            return configured
        raise Refusal(f"LMS_CLI={configured} is not an executable file")
    return shutil.which("lms") or shutil.which("lms.exe")


def manual_instructions(missing: list[str]) -> str:
    cmds = "\n".join(f"      lms {' '.join(load_args(m))}" for m in missing)
    return (
        "  Load them yourself, then re-run:\n"
        f"{cmds}\n"
        f"  (or in the LM Studio app: context length {DEMO_CONTEXT_LENGTH}, GPU offload max).\n"
        "  To let this step load them, point LMS_CLI at the lms CLI\n"
        "  (e.g. LMS_CLI=/mnt/c/Users/<you>/.lmstudio/bin/lms.exe under WSL) or put lms on PATH."
    )


def _settle_after_load(demo: DemoSet) -> dict[str, dict]:
    """Re-read until no demo model reports `loading` (whether LM Studio's API can still
    say so after `lms load` returns is unknown), up to SETTLE_S; a model still `loading`
    then is refused by check_refusals. Post-load only: before a load, `loading` means
    someone else's load is in flight, and that stays an immediate refusal."""
    deadline = time.monotonic() + SETTLE_S
    while True:
        after = loaded_models(read_models(demo.api_root))
        busy = [mid for mid in demo.all_ids if after.get(mid, {}).get("state") == "loading"]
        if not busy or time.monotonic() >= deadline:
            return after
        _say("waiting for " + ", ".join(busy) + " to finish loading")
        time.sleep(SETTLE_INTERVAL_S)


def ensure(demo: DemoSet) -> int:
    _say(f"demo models: chat={', '.join(demo.chat)}  embedding={demo.embedding}  "
         f"(LM Studio at {demo.api_root})")
    loaded = loaded_models(read_models(demo.api_root))
    check_refusals(demo, loaded)

    missing = [mid for mid in demo.all_ids if mid not in loaded]
    if not missing:
        _say("both demo models already loaded with the demo settings — nothing to do")
        return 0

    _say("not loaded: " + ", ".join(missing))
    cli = find_cli()
    if cli is None:
        raise Refusal("the lms CLI was not found (LMS_CLI unset, no lms/lms.exe on PATH).\n"
                      + manual_instructions(missing))

    for mid in missing:
        args = [cli, *load_args(mid)]
        _say("loading (can take a minute): " + " ".join(args))
        try:
            proc = subprocess.run(args, capture_output=True, text=True, timeout=LOAD_TIMEOUT_S)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise Refusal(f"`lms load {mid}` did not complete: {exc}\n" + manual_instructions([mid]))
        if proc.returncode != 0:
            tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-5:])
            raise Refusal(f"`lms load {mid}` exited {proc.returncode}:\n{tail}\n"
                          + manual_instructions([mid]))

    # Re-read: catches a CLI wired to a different LM Studio than the server's URL, and a
    # load that evicted something.
    after = _settle_after_load(demo)
    still_missing = [mid for mid in demo.all_ids if mid not in after]
    if still_missing:
        raise Refusal(
            "after loading, the LM Studio API at "
            f"{demo.api_root} still does not list as loaded: {', '.join(still_missing)}.\n"
            "  Is the lms CLI driving the same LM Studio instance as that URL?"
        )
    check_refusals(demo, after)
    _say("loaded: " + ", ".join(missing) + " — demo models ready")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ensure LM Studio has the storefront demo's models loaded (never unloads).",
    )
    parser.add_argument("--print-set", action="store_true",
                        help="print the derived demo model set and LM Studio URL, then exit")
    args = parser.parse_args(argv)
    try:
        demo = derive_demo_set()
        if args.print_set:
            print(json.dumps({"api_root": demo.api_root, "chat": list(demo.chat),
                              "embedding": demo.embedding}))
            return 0
        return ensure(demo)
    except Refusal as exc:
        print(f"{PREFIX} ERROR: {exc}", file=sys.stderr, flush=True)
        return exc.exit_code


if __name__ == "__main__":
    sys.exit(main())
