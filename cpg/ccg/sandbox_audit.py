"""The ``sandboxAuditIds`` check (``-egress-spec3.md`` §5.3-§5.4): one function, no broker dependency.

``check_sandbox_ids(header, audit_dir) -> list[str]`` (``[]`` = valid) is called at assembly (P1-18) and again at load (P1-19). A
bundle is refused unless, for **every** id in ``header.sandboxAuditIds``:

1. ``<audit_dir>/sandbox.jsonl`` holds exactly one record whose ``auditId`` is that id (zero or several is invalid);
2. that record validates against ``schemas/sandbox-record-v1.schema.json`` (lib-launch records are rejected there);
3. it has ``refused: false`` and ``exitCode: 0``;
4. its ``egressMode`` equals ``header.egressMode``, and in P1 both are ``offline``.

An empty list is valid exactly when ``header.unit.lang`` is in :data:`HOST_LANGS` (a host extractor runs on the host: no sandbox
invocation exists) and the mode is ``offline``; ``[]`` from this function means *valid*, never *attested* (:func:`attestation` says what a valid header is). The check is presence-only: it proves a successful offline invocation with that id exists, not that the bundle's inputs came from it
(the record carries no output digest). Each error names the id (a validated UUIDv4) or the list index, and the failed condition; never a header string of unchecked shape, never record content.

**Exit 124.** The runner's ``124`` means "an indexer ran past its wall-clock cap (``SBX_INDEXER_WALL_S``, 1800 s) **or** a tool that
exited 124 itself" (``cpg/sandbox/README.md``); a boundary race can also record 124 for a run that had just succeeded. Every 124 is
refused (it is not exit 0); the error compares ``durationMs`` with the cap so a reader can tell which it most likely was.
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from functools import cache
from pathlib import Path

import jsonschema

from ccg.bundle import HOST_LANGS

SCHEMA_PATH = Path(__file__).parent / "schemas" / "sandbox-record-v1.schema.json"
INDEXER_WALL_CAP_MS = 1800 * 1000
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")


@cache
def _validator() -> jsonschema.protocols.Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    cls = jsonschema.validators.validator_for(schema)
    cls.check_schema(schema)
    return cls(schema)


def record_errors(record: object) -> list[str]:
    """Schema errors of one sandbox record as ``path: validator`` strings (no instance values)."""
    return sorted("/".join(map(str, e.absolute_path)) + ": " + e.validator for e in _validator().iter_errors(record))


def default_audit_dir() -> Path:
    """``CCG_AUDIT_DIR`` when set, else ``<directory of the ccg package>/../.audit`` (egress spec §5.1)."""
    env = os.environ.get("CCG_AUDIT_DIR")
    return Path(env) if env else Path(__file__).resolve().parent.parent / ".audit"


def _records(path: Path, wanted: set[str]) -> dict[str, list[object]]:
    """Records whose ``auditId`` is wanted, by id; a trailing partial line and an unparseable line are skipped."""
    out: dict[str, list[object]] = {i: [] for i in wanted}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    for line in text.splitlines():
        if not any(i in line for i in wanted):
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict) and isinstance(rec.get("auditId"), str) and rec["auditId"] in out:
            out[rec["auditId"]].append(rec)
    return out


def check_sandbox_ids(header: Mapping, audit_dir: Path) -> list[str]:
    ids = header.get("sandboxAuditIds") if isinstance(header, Mapping) else None
    lang = header.get("unit", {}).get("lang") if isinstance(header, Mapping) and isinstance(header.get("unit"), Mapping) else None
    if isinstance(ids, list) and not ids and lang in HOST_LANGS:
        return [] if header.get("egressMode") == "offline" else ["egressMode: the header's mode is not 'offline' (the only mode P1 accepts)"]
    if not isinstance(ids, list) or not ids:
        return ["sandboxAuditIds: none named (only a host-extractor unit may name none)"]
    errors: list[str] = []
    if header.get("egressMode") != "offline":
        errors.append("egressMode: the header's mode is not 'offline' (the only mode P1 accepts)")
    wanted = {i for i in ids if isinstance(i, str) and _UUID.fullmatch(i)}
    # value-free: the index and the rule, never the header's string (a malformed entry may be anything, a secret included)
    errors += [f"sandboxAuditIds[{n}]: not a lowercase UUIDv4" for n, i in enumerate(ids)
               if not (isinstance(i, str) and _UUID.fullmatch(i))][:8]
    found = _records(Path(audit_dir) / "sandbox.jsonl", wanted)
    for i in sorted(wanted):
        recs = found[i]
        if len(recs) != 1:
            errors.append(f"auditId {i}: {len(recs)} records in sandbox.jsonl, exactly one required")
            continue
        rec = recs[0]
        if record_errors(rec):
            errors.append(f"auditId {i}: the record does not validate against sandbox-record-v1")
        elif rec["refused"] or rec["exitCode"] != 0:
            detail = ""
            if rec["exitCode"] == 124 and not rec["refused"]:
                fired = rec["durationMs"] >= INDEXER_WALL_CAP_MS
                detail = (f" (124 is the watchdog or the tool's own exit: durationMs {rec['durationMs']} is "
                          f"{'at or above' if fired else 'below'} the {INDEXER_WALL_CAP_MS} ms cap)")
            errors.append(f"auditId {i}: refused={rec['refused']}, exitCode={rec['exitCode']}, a successful invocation is required{detail}")
        elif rec["egressMode"] != header.get("egressMode"):
            errors.append(f"auditId {i}: the record's egressMode differs from the header's")
    return errors


def attestation(header: Mapping, audit_dir: Path) -> str:
    """``invalid`` (``check_sandbox_ids`` returned errors), else ``host-unattested`` for any valid header of a ``HOST_LANGS`` unit and
    ``sandboxed`` for a valid code-language header. The language decides, not the list: a host extractor never runs in the sandbox, so
    ids a host header names (checked as for any bundle) are not an attestation that a sandboxed run produced its inputs."""
    try:
        if check_sandbox_ids(header, audit_dir):
            return "invalid"
        lang = header["unit"]["lang"]
        if not isinstance(lang, str):
            return "invalid"
        return "host-unattested" if lang in HOST_LANGS else "sandboxed"
    except Exception:                                                # noqa: BLE001 - total and fail-closed: a garbage header is invalid
        return "invalid"
