"""KB mirror freshness: the REST writes, the read-only observation, the per-section engine (plan 3.5).

A KB file is mirrored into `ws:agent-team` one document per `##` heading (mode `whole`) or per top-level
bullet (`<!-- kb-split: bullets -->`). The mapping (`:KbFile`/`:KbSection`, graph note 2.4) lives in the
mirror graph; documents live in falkor-chat. **Writes** go through falkor-chat's REST (`KbHttp`, stdlib
`urllib`, producer `kb-mirror`); **observation** (embed landed, current version, normalised hash) is a
read-only graph read of `$COORDGRAPH_KB_GRAPH` (`Observer`, note 6.4, coupling R10, pinned by the live
contract test). This module never imports `wilib` or `identity`.

Concurrency unit = the section. Every mapping change runs between a lease acquire and a release (a
compare-and-set in the mirror graph); every mapping write is fenced on the holder's token, and rules L1-L3
(note 6.4) protect falkor-chat's side effects: renew before every call, delete your own unrecorded POST,
delete old ids only under a freshly renewed lease. A lost lease means: stop, make no further falkor-chat
call for that section, report it *pending*.

Append before delete: a changed section POSTs its new documents, records them as `pending*`, waits (bounded
by the budget) for every new chunk to carry an embedding, then deletes the old ids and swaps. Over budget the
section stays `pending` (old claim still served and flagged by `kb-status`) and `finish` completes it.
Unchanged claims (same normalised text) are kept, not re-posted.

Additions to note 2.4 (reported to its owner): `mode`, `needsResplit` on a section; `pendingClaimTitles`,
`pendingKeepHashes`, `pendingMode` while pending (the swap keeps unchanged old claims, so the pending set
is the *new* documents only).
"""
from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from .config import LEASE_TTL_S
from .keys import KeyHygieneError, safe_key
from .parse.kb import _BULLET, Claim, KbDoc, Section, claim_hash, duplicate_claims
from .store import StoreError

PRODUCER = "kb-mirror"
TTL_MS = LEASE_TTL_S * 1000
FINISH_CAP_MS = 15 * 60 * 1000  # plan 3.5: per-section cap of the detached `kb-finish`
POLL_S = 0.5
TITLE_LIMIT = 200  # falkor-chat's `ingest_document` title bound

# --- the mirror graph's statements (graph note 6.4, verbatim except the marked additions) -------------

ACQUIRE = """MERGE (ks:KbSection {key: $key})
ON CREATE SET ks.path = $path, ks.heading = $heading, ks.state = 'pending',
              ks.pendingSince = timestamp()
WITH ks
WHERE ks.leaseUntil IS NULL OR ks.leaseUntil < timestamp()
SET ks.leaseToken = $token, ks.leaseUntil = timestamp() + $ttlMs
RETURN ks.leaseToken AS token, ks.leaseUntil AS leaseUntil"""

RELEASE = """MATCH (ks:KbSection {key: $key})
WHERE ks.leaseToken = $token
SET ks.leaseToken = null, ks.leaseUntil = null
RETURN count(ks) AS released"""

RENEW = """MATCH (ks:KbSection {key: $key})
WHERE ks.leaseToken = $token
SET ks.leaseUntil = timestamp() + $ttlMs
RETURN count(ks) AS renewed"""

_FENCE = "MATCH (ks:KbSection {key: $key})\nWHERE ks.leaseToken = $token\n"
_CLEAR_PENDING = ("ks.pendingDocumentIds = null, ks.pendingClaimHashes = null, ks.pendingClaimTitles = null, "
                  "ks.pendingKeepHashes = null, ks.pendingMode = null, ks.pendingSectionHash = null")

# the pending marker, written before any POST; carries the renew (note 6.4 L1)
MARK = (_FENCE + "SET ks.state = 'pending', ks.pendingSince = timestamp(), ks.pendingSectionHash = $sectionHash, "
        "ks.pendingKeepHashes = $keep, ks.pendingMode = $mode, ks.leaseUntil = timestamp() + $ttlMs\n"
        "RETURN count(ks) AS n")
RECORD = (_FENCE + "SET ks.pendingDocumentIds = $ids, ks.pendingClaimHashes = $hashes, "
          "ks.pendingClaimTitles = $titles\nRETURN count(ks) AS n")
CANCEL = _FENCE + f"SET {_CLEAR_PENDING}\nRETURN count(ks) AS n"
# give up: the old ids stay mapped and served; the old `sectionHash` is kept, so `kb-status` flags it stale
GIVE_UP = (_FENCE + f"SET {_CLEAR_PENDING}, ks.state = 'synced', ks.pendingSince = null, "
           "ks.leaseToken = null, ks.leaseUntil = null\nRETURN count(ks) AS n")
SWAP = (_FENCE + "SET ks.documentIds = $ids, ks.claimHashes = $hashes, ks.claimTitles = $titles, "
        "ks.sectionHash = $sectionHash, ks.mode = $mode, ks.needsResplit = $needsResplit, ks.heading = $heading, "
        "ks.ord = $ord, ks.state = 'synced', ks.syncedAt = timestamp(), "
        f"{_CLEAR_PENDING}, ks.pendingSince = null, ks.leaseToken = null, ks.leaseUntil = null\nRETURN count(ks) AS n")
REMOVE = _FENCE + "DETACH DELETE ks\nRETURN count(ks) AS n"
READ_ONE = "MATCH (ks:KbSection {key: $key}) RETURN properties(ks) AS p, timestamp() AS now"
READ_ALL = "MATCH (ks:KbSection) RETURN properties(ks) AS p, timestamp() AS now"
READ_FILES = "MATCH (f:KbFile) RETURN f.path AS path, f.sha256 AS sha256, f.syncedAt AS syncedAt"
UPSERT_FILE = ("MERGE (f:KbFile {path: $path}) SET f.sha256 = $sha256, f.syncedAt = timestamp() "
               "WITH f MATCH (ks:KbSection {path: $path}) MERGE (f)-[:HAS_SECTION]->(ks) RETURN count(ks) AS n")

# --- the observation reads of falkor-chat's graph (note 6.4; the schema coupling of R10) ----------------

OBSERVE = """UNWIND $ids AS id
OPTIONAL MATCH (d:Document {documentId: id})
OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
RETURN id, d.textNormalizedHash AS hash, d.status AS status, d.currentVersion AS current,
       count(c) AS chunks, count(c.embedding) AS embedded"""
DESCRIBE = ("OPTIONAL MATCH (d:Document {documentId: $id}) OPTIONAL MATCH (d)-[:INGESTED_BY]->(a) "
            "RETURN d.textNormalizedHash AS hash, a.agentId AS producer")
BY_HASH = ("UNWIND $hashes AS h MATCH (d:Document {textNormalizedHash: h, currentVersion: true}) "
           "RETURN h AS hash, d.documentId AS id")
PRODUCED = ("MATCH (d:Document)-[:INGESTED_BY]->(a:Agent {agentId: $agent}) "
            "RETURN d.documentId AS id, d.textNormalizedHash AS hash, d.currentVersion AS current, d.title AS title")


class KbHttpError(RuntimeError):
    """A falkor-chat REST call that did not return a usable answer. `timed_out` means the client gave up
    while the server may still have committed: the outcome is unknown (review P4-m1)."""

    def __init__(self, message: str, *, status: int | None = None, timed_out: bool = False):
        super().__init__(message)
        self.status = status
        self.timed_out = timed_out


class LeaseLost(Exception):
    """A fenced write or renew matched no row: another actor holds the section now."""


class KbHttp:
    """falkor-chat's REST surface for the KB mirror: `POST /documents`, `DELETE /documents/{id}`, `GET /health`."""

    def __init__(self, base_url: str, timeout_s: float, producer: str = PRODUCER):
        self.base = base_url.rstrip("/")
        self.timeout = timeout_s
        self.producer = producer

    def _call(self, method: str, path: str, body: dict | None = None, ok404: bool = False) -> dict:
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404 and ok404:
                return {}
            detail = exc.read()[:200].decode("utf-8", "replace")
            raise KbHttpError(f"{method} {path}: HTTP {exc.code} {detail}", status=exc.code) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise KbHttpError(f"{method} {path}: client timeout after {self.timeout:g}s", timed_out=True) from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise KbHttpError(f"{method} {path}: client timeout after {self.timeout:g}s", timed_out=True) from exc
            raise KbHttpError(f"{method} {path}: {exc.reason}") from exc
        except OSError as exc:
            raise KbHttpError(f"{method} {path}: {exc}") from exc
        try:
            return json.loads(raw or b"{}")
        except ValueError as exc:
            raise KbHttpError(f"{method} {path}: response is not JSON") from exc

    def health(self) -> None:
        self._call("GET", "/health")

    def post_document(self, title: str, text: str) -> dict:
        receipt = self._call("POST", "/documents", {"text": text, "title": title, "sourceFormat": "markdown",
                                                    "producedBy": self.producer})
        if not isinstance(receipt, dict) or not receipt.get("documentId"):
            raise KbHttpError(f"POST /documents: receipt without documentId: {receipt!r}")
        return receipt

    def delete_document(self, document_id: str) -> None:
        """Delete by id; a 404 counts as done (note 6.4)."""
        self._call("DELETE", f"/documents/{document_id}", ok404=True)


@dataclass(frozen=True)
class Obs:
    """What the observation read says about one document id."""

    id: str
    hash: str | None
    status: str | None
    current: bool | None
    chunks: int
    embedded: int

    @property
    def exists(self) -> bool:
        return self.status is not None

    @property
    def ready(self) -> bool:
        return self.exists and self.chunks > 0 and self.embedded == self.chunks

    @property
    def failed(self) -> bool:
        return self.status == "failed"


class Observer:
    """Read-only reads of falkor-chat's workspace graph (`graph` is a falkordb graph, used via `ro_query`)."""

    def __init__(self, graph):
        self.graph = graph

    def _rows(self, statement: str, params: dict) -> list:
        try:
            return self.graph.ro_query(statement, params).result_set
        except Exception as exc:  # noqa: BLE001 - an absent key, a schema change, a lost connection
            raise StoreError(f"KB graph read failed: {exc}") from exc

    def observe(self, ids: list[str]) -> dict[str, Obs]:
        if not ids:
            return {}
        return {r[0]: Obs(r[0], r[1], r[2], r[3], int(r[4] or 0), int(r[5] or 0))
                for r in self._rows(OBSERVE, {"ids": list(ids)})}

    def describe(self, document_id: str) -> tuple[str | None, str | None]:
        """`(textNormalizedHash, producing agent id)` of one document; `(None, None)` when it does not exist."""
        rows = self._rows(DESCRIBE, {"id": document_id})
        return (rows[0][0], rows[0][1]) if rows else (None, None)

    def current_by_hash(self, hashes: list[str]) -> dict[str, str]:
        """`{hash: documentId}` of current documents carrying one of `hashes`."""
        if not hashes:
            return {}
        return {r[0]: r[1] for r in self._rows(BY_HASH, {"hashes": list(hashes)})}

    def produced_by(self, agent: str) -> list[tuple[str, str | None, bool | None, str | None]]:
        return [tuple(r) for r in self._rows(PRODUCED, {"agent": agent})]


# --- model --------------------------------------------------------------------------------------------

def claim_title(path: str, section: Section, claim: Claim) -> str:
    """Recorded title (agent-maintenance 5 step 5): the heading for a whole claim, `<family> - <lead>` for a
    split sibling; shortened to falkor-chat's bound by cutting the claim title, never the family slug."""
    if section.mode == "whole":
        return section.heading[:TITLE_LIMIT]
    family = path.rsplit("/", 1)[-1].removesuffix(".md")
    prefix = f"{family} — "
    return prefix + claim.title[: max(0, TITLE_LIMIT - len(prefix))]


def section_key(path: str, section: Section) -> str:
    return safe_key(path, section.heading_key)


def _live_keys(docs: Iterable[KbDoc]) -> set[str]:
    """Keys of every section that has one. A heading whose key is unsafe has none (its own sync fails, per
    section, `sync_section`); it must never abort the run that is only collecting facts."""
    keys: set[str] = set()
    for d in docs:
        for s in d.sections:
            try:
                keys.add(section_key(d.path, s))
            except KeyHygieneError:
                continue
    return keys


def _stripped_hash(claim: Claim) -> str | None:
    """The hash of a bullets claim without its list marker: how the claims stored before the marker
    became part of the text (the 17 of `falkordb-reference.md`) are hashed."""
    stripped = _BULLET.sub("", claim.text, count=1)
    return claim_hash(stripped) if stripped != claim.text else None


@dataclass(frozen=True)
class KbOutcome:
    path: str
    heading: str
    status: str  # synced | unchanged | pending | refused | failed | removed
    detail: str = ""

    @property
    def incomplete(self) -> bool:
        return self.status in ("pending", "refused", "failed")

    def render(self) -> str:
        return f"{self.status}\t{self.path} § {self.heading}" + (f"\t{self.detail}" if self.detail else "")


@dataclass
class Context:
    """Run-level facts every section decision needs: who owns each claim hash, which sections vanished."""

    live_keys: set[str]
    owners: dict[str, tuple[str, str, str]]  # claim hash -> (key, path, heading)
    vanished_ids: set[str]  # document ids of sections no longer in any registered file
    mapped_ids: set[str]  # every id some section maps (documentIds + pendingDocumentIds)
    released_ids: set[str] = field(default_factory=set)  # ids a live section maps whose claim its file text dropped

    @classmethod
    def build(cls, docs: Iterable[KbDoc], rows: list[dict]) -> Context:
        docs = list(docs)
        live = _live_keys(docs)
        wanted: dict[str, set[str]] = {}  # live key -> claim hashes its file text carries now
        for d in docs:
            for s in d.sections:
                try:
                    wanted.setdefault(section_key(d.path, s), set()).update(c.claim_hash for c in s.claims)
                except KeyHygieneError:
                    continue
        owners: dict[str, tuple[str, str, str]] = {}
        vanished: set[str] = set()
        mapped: set[str] = set()
        released: set[str] = set()
        for row in rows:
            ids = [*(row.get("documentIds") or []), *(row.get("pendingDocumentIds") or [])]
            mapped.update(ids)
            if row["key"] in live:
                # the file wins over the old mapping: a claim the section's text no longer carries is not owned
                # by it, so headings exchanging claims in one edit do not each block the other
                for i, h in zip(row.get("documentIds") or [], row.get("claimHashes") or [], strict=False):
                    if h in wanted[row["key"]]:
                        owners.setdefault(h, (row["key"], row.get("path", ""), row.get("heading", "")))
                    else:
                        released.add(i)
            else:
                vanished.update(row.get("documentIds") or [])
                vanished.update(row.get("pendingDocumentIds") or [])
        for d in docs:
            for s in d.sections:
                try:
                    key = section_key(d.path, s)
                except KeyHygieneError:
                    continue
                for c in s.claims:
                    owners.setdefault(c.claim_hash, (key, d.path, s.heading))
        return cls(live, owners, vanished, mapped, released)

    def update(self, key: str, path: str, heading: str, hashes: list[str]) -> None:
        """After a swap: the section owns exactly `hashes` now."""
        for h in [h for h, (k, _, _) in self.owners.items() if k == key]:
            del self.owners[h]
        for h in hashes:
            self.owners.setdefault(h, (key, path, heading))

    def refusal(self, key: str, section: Section) -> str | None:
        hashes = [c.claim_hash for c in section.claims]
        if len(set(hashes)) != len(hashes):
            return f"two claims of '{section.heading}' have the same text"
        for h in hashes:
            owner = self.owners.get(h)
            if owner and owner[0] != key:
                return (f"claim text duplicates '{owner[2]}' ({owner[1]}); falkor-chat would supersede one with "
                        "the other: dedupe in the file")
        return None


@dataclass
class Plan:
    keep: list[tuple[str, str, str]]  # (id, hash, title) of old claims whose text is unchanged
    new: list[Claim]
    remove: list[str]  # old ids no longer wanted (or untrustworthy)
    resplit: bool


def make_plan(section: Section, row: dict, bad_ids: frozenset[str] = frozenset()) -> Plan:
    ids = row.get("documentIds") or []
    hashes = row.get("claimHashes") or []
    titles = list(row.get("claimTitles") or [])
    titles += [""] * (len(ids) - len(titles))
    want = {c.claim_hash for c in section.claims}
    keep: list[tuple[str, str, str]] = []
    remove: list[str] = []
    kept: set[str] = set()
    for i, h, t in zip(ids, hashes, titles, strict=False):
        if h in want and h not in kept and i not in bad_ids:
            keep.append((i, h, t))
            kept.add(h)
        else:
            remove.append(i)
    new = [c for c in section.claims if c.claim_hash not in kept]
    return Plan(keep, new, remove, resplit=section.mode == "whole" and len(ids) > 1)


# --- the engine ---------------------------------------------------------------------------------------

class KbSync:
    """Per-section sync, finish and reconcile over a mirror `graph`, a `KbHttp` and an `Observer`.

    `sleep`/`monotonic` are injectable so a test never waits; `ttl_ms` and `cap_ms` likewise.
    """

    def __init__(self, graph, http, observer, *, budget_s: float = 10.0, poll_s: float = POLL_S,
                 ttl_ms: int = TTL_MS, cap_ms: int = FINISH_CAP_MS, sleep: Callable[[float], None] = time.sleep,
                 monotonic: Callable[[], float] = time.monotonic, token: Callable[[], str] | None = None):
        self.graph, self.http, self.observer = graph, http, observer
        self.budget_s, self.poll_s, self.ttl_ms, self.cap_ms = budget_s, poll_s, ttl_ms, cap_ms
        self.sleep, self.monotonic = sleep, monotonic
        self.new_token = token or (lambda: uuid.uuid4().hex)

    # -- mirror graph ----------------------------------------------------------------------------
    def _q(self, statement: str, **params) -> list:
        try:
            return self.graph.query(statement, params).result_set
        except Exception as exc:  # noqa: BLE001 - falkordb ResponseError and friends
            raise StoreError(f"mirror graph: {exc}") from exc

    def _fenced(self, statement: str, key: str, token: str, **params) -> bool:
        rows = self._q(statement, key=key, token=token, **params)
        return bool(rows) and bool(rows[0][0])

    def _acquire(self, key: str, path: str, heading: str) -> str | None:
        token = self.new_token()
        return token if self._q(ACQUIRE, key=key, path=path, heading=heading, token=token, ttlMs=self.ttl_ms) else None

    def _renew(self, key: str, token: str) -> None:
        """L1: before every falkor-chat call. 0 rows: stop before calling."""
        if not self._fenced(RENEW, key, token, ttlMs=self.ttl_ms):
            raise LeaseLost("lease expired or taken over before a falkor-chat call")

    def _delete(self, key: str, token: str, document_id: str) -> None:
        """One falkor-chat DELETE, each under its own renew (L1): N deletes may outlast the lease's TTL."""
        self._renew(key, token)
        self.http.delete_document(document_id)

    def _release(self, key: str, token: str) -> None:
        try:
            self._fenced(RELEASE, key, token)
        except StoreError:
            pass  # the lease expires by itself

    def _read(self, key: str) -> tuple[dict, int]:
        rows = self._q(READ_ONE, key=key)
        return (dict(rows[0][0]), int(rows[0][1])) if rows else ({}, 0)

    def read_all(self) -> tuple[list[dict], int]:
        rows = self._q(READ_ALL)
        return [dict(r[0]) for r in rows], (int(rows[0][1]) if rows else 0)

    # -- one section -----------------------------------------------------------------------------
    def sync_section(self, path: str, section: Section, ctx: Context, *, force: bool = False,
                     bad_ids: frozenset[str] = frozenset()) -> KbOutcome:
        def out(status: str, detail: str = "") -> KbOutcome:
            return KbOutcome(path, section.heading, status, detail)

        try:
            key = section_key(path, section)
            token = self._acquire(key, path, section.heading)
        except (KeyHygieneError, StoreError) as exc:
            return out("failed", str(exc))
        if token is None:
            return out("pending", "section lease is held by another actor; retry in 30 s")
        try:
            return self._sync_locked(key, token, path, section, ctx, force, bad_ids, out)
        except LeaseLost as exc:
            return out("pending", f"{exc}; no further falkor-chat call made for this section")
        except (StoreError, KbHttpError) as exc:
            return out("failed", str(exc))
        finally:
            self._release(key, token)

    def _sync_locked(self, key, token, path, section, ctx, force, bad_ids, out) -> KbOutcome:
        row, _ = self._read(key)
        if row.get("pendingDocumentIds") and row.get("pendingSectionHash") == section.section_hash \
                and row.get("pendingMode") == section.mode:  # the same edit, still in flight: `finish` owns it
            return out("pending", "this exact text is already submitted and embedding; left to `kb-finish`")
        if row.get("pendingDocumentIds"):  # cancel-on-re-edit: delete before clearing (note 6.4)
            for document_id in row["pendingDocumentIds"]:
                self._delete(key, token, document_id)
            if not self._fenced(CANCEL, key, token):
                raise LeaseLost("lease lost clearing a cancelled submission")
            for name in ("pendingDocumentIds", "pendingClaimHashes", "pendingClaimTitles", "pendingKeepHashes",
                         "pendingMode", "pendingSectionHash"):
                row.pop(name, None)
        if not force and row.get("state") == "synced" and row.get("sectionHash") == section.section_hash \
                and row.get("mode") == section.mode:
            failed = self._failed_ids(row)  # one read under the lease; `failed` is terminal, so never stale
            if not failed:
                return out("unchanged")
            bad_ids = bad_ids | failed  # a failed document never recovers: re-submit its claim (not `processing`)
        reason = ctx.refusal(key, section)
        if reason:
            return out("refused", reason)
        plan = make_plan(section, row, bad_ids)
        if not self._fenced(MARK, key, token, sectionHash=section.section_hash, mode=section.mode,
                            keep=[h for _, h, _ in plan.keep], ttlMs=self.ttl_ms):
            raise LeaseLost("lease lost writing the pending marker")
        posted: list[tuple[str, str, str]] = []
        for claim in plan.new:
            self._renew(key, token)
            try:
                receipt = self.http.post_document(claim_title(path, section, claim), claim.text)
            except KbHttpError as exc:
                note = ("; the server may still have committed it (outcome unknown): a retry supersedes it"
                        if exc.timed_out else "")
                return self._abort(key, token, posted, out, f"{exc}{note}")
            new_id = receipt["documentId"]
            posted.append((new_id, claim.claim_hash, claim_title(path, section, claim)))
            problem = self._check_receipt(key, token, receipt, claim, row, ctx)
            if problem:
                return self._abort(key, token, posted, out, problem)
            if not self._fenced(RECORD, key, token, ids=[p[0] for p in posted], hashes=[p[1] for p in posted],
                                titles=[p[2] for p in posted]):
                self.http.delete_document(new_id)  # L2: clean up your own POST
                raise LeaseLost("lease lost recording a submission (its document was deleted)")
        if posted:
            wait = self._wait_embedded(key, token, [p[0] for p in posted])
            if wait == "failed":
                return self._abort(key, token, posted, out, "a submitted document failed or vanished before embedding")
            if wait == "budget":
                return out("pending", "submitted, not yet searchable; old version served and flagged")
        self._swap(key, token, path, section, plan, posted, row)
        ctx.update(key, path, section.heading, [c.claim_hash for c in section.claims])
        return out("synced", f"{len(posted)} new, {len(plan.keep)} kept, {len(plan.remove)} removed")

    def _failed_ids(self, row: dict) -> frozenset[str]:
        """The section's mapped documents falkor-chat marked `failed` (its terminal status), one batched read."""
        ids = row.get("documentIds") or []
        obs = self.observer.observe(ids)
        return frozenset(i for i in ids if obs.get(i) is not None and obs[i].failed)

    def _check_receipt(self, key, token, receipt, claim, row, ctx) -> str | None:
        """The receipt backstop (m8, P4-m1): falkor-chat auto-supersedes a current document with the same
        normalised text. Acceptable victims: this section's own old documents, a vanished section's, one whose
        section's file text no longer carries that claim (headings exchanging claims), or an
        unmapped `kb-mirror` document of this very text (our own POST whose answer we never saw)."""
        if not receipt.get("autoSuperseded"):
            return None
        victim = receipt.get("supersededDocumentId")
        if victim in (row.get("documentIds") or []) or victim in ctx.vanished_ids or victim in ctx.released_ids:
            return None
        digest, producer = self.observer.describe(victim) if victim else (None, None)
        _, mapped = self._mapped_ids()
        if producer == PRODUCER and digest == claim.claim_hash and victim not in mapped:
            self._renew(key, token)
            self.http.delete_document(victim)  # the superseded twin of our own timed-out POST
            return None
        return f"POST superseded document {victim}, which this section does not own: report, never guess"

    def _mapped_ids(self) -> tuple[list[dict], set[str]]:
        rows, _ = self.read_all()
        return rows, {i for r in rows for i in [*(r.get("documentIds") or []), *(r.get("pendingDocumentIds") or [])]}

    def _abort(self, key, token, posted, out, reason) -> KbOutcome:
        """Delete every document this run submitted, clear the pending fields; old claim stays served."""
        for document_id, _, _ in posted:
            self._delete(key, token, document_id)
        if not self._fenced(GIVE_UP, key, token):
            raise LeaseLost("lease lost giving up")
        return out("failed", reason)

    def _wait_embedded(self, key, token, ids: list[str]) -> str:
        deadline = self.monotonic() + self.budget_s
        while True:
            obs = self.observer.observe(ids)
            seen = [obs.get(i) for i in ids]
            if any(o is None or not o.exists or o.failed for o in seen):
                return "failed"
            if all(o.ready for o in seen):
                return "ready"
            if self.monotonic() >= deadline:
                return "budget"
            self._renew(key, token)
            self.sleep(self.poll_s)

    def _swap(self, key, token, path, section, plan: Plan, posted, row) -> None:
        """L3: renew, delete the old ids, then the fenced swap with the release folded in."""
        for document_id in plan.remove:
            self._delete(key, token, document_id)
        by_hash = {h: (i, t) for i, h, t in plan.keep}
        by_hash.update({h: (i, t) for i, h, t in posted})
        wanted = [c.claim_hash for c in section.claims]
        ok = self._fenced(SWAP, key, token, ids=[by_hash[h][0] for h in wanted], hashes=wanted,
                          titles=[by_hash[h][1] for h in wanted], sectionHash=section.section_hash,
                          mode=section.mode, needsResplit=True if plan.resplit else None, heading=section.heading,
                          ord=section.ord)
        if not ok:
            raise LeaseLost("lease lost during the swap")

    # -- one file --------------------------------------------------------------------------------
    def sync_file(self, doc: KbDoc, ctx: Context, rows: list[dict]) -> list[KbOutcome]:
        """Every section of `doc`, then the sections that vanished from it. One section's failure never stops
        the others; `:KbFile` is stamped only when nothing is pending, refused or failed."""
        outcomes = [self.sync_section(doc.path, s, ctx) for s in doc.sections if s.claims]
        live = _live_keys([doc])
        for row in rows:
            if row.get("path") == doc.path and row["key"] not in live:
                outcomes.append(self.remove_section(row))
        if not any(o.incomplete for o in outcomes):
            try:
                self._q(UPSERT_FILE, path=doc.path, sha256=doc.sha256)
            except StoreError as exc:
                outcomes.append(KbOutcome(doc.path, "(file record)", "failed", str(exc)))
        return outcomes

    def remove_section(self, row: dict) -> KbOutcome:
        """A heading that left the file: delete its documents, drop the record (note: removed section)."""
        key, path, heading = row["key"], row.get("path", ""), row.get("heading", row["key"])
        try:
            token = self._acquire(key, path, heading)
        except StoreError as exc:
            return KbOutcome(path, heading, "failed", str(exc))
        if token is None:
            return KbOutcome(path, heading, "pending", "section lease is held by another actor; retry in 30 s")
        try:
            current, _ = self._read(key)
            ids = [*(current.get("documentIds") or []), *(current.get("pendingDocumentIds") or [])]
            for document_id in ids:
                self._delete(key, token, document_id)
            if not self._fenced(REMOVE, key, token):
                raise LeaseLost("lease lost dropping the record")
            return KbOutcome(path, heading, "removed", f"{len(ids)} document(s) deleted")
        except LeaseLost as exc:
            return KbOutcome(path, heading, "pending", str(exc))
        except (StoreError, KbHttpError) as exc:
            self._release(key, token)
            return KbOutcome(path, heading, "failed", str(exc))

    # -- finish: complete or give up the pending sections ---------------------------------------
    def finish(self, *, once: bool = False) -> list[KbOutcome]:
        """Loop over `pending` sections, one poll-and-swap step each under that section's lease only."""
        results: dict[str, KbOutcome] = {}
        while True:
            rows, _ = self.read_all()
            waiting = 0
            for row in rows:
                if row.get("state") != "pending":
                    continue
                result = self._finish_one(row)
                results[row["key"]] = result
                waiting += result.status == "pending"
            if once or not waiting:
                return list(results.values())
            self.sleep(self.poll_s * 10)

    def _finish_one(self, row: dict) -> KbOutcome:
        key, path, heading = row["key"], row.get("path", ""), row.get("heading", row["key"])

        def out(status: str, detail: str = "") -> KbOutcome:
            return KbOutcome(path, heading, status, detail)

        try:
            token = self._acquire(key, path, heading)
        except StoreError as exc:
            return out("failed", str(exc))
        if token is None:
            return out("pending", "section lease is held by another actor")
        try:
            row, now = self._read(key)
            if row.get("state") != "pending":
                return out("unchanged")
            ids = row.get("pendingDocumentIds") or []
            capped = now - int(row.get("pendingSince") or now) > self.cap_ms
            obs = self.observer.observe(ids)
            seen = [obs.get(i) for i in ids]
            bad = any(o is None or not o.exists or o.failed for o in seen)
            if ids and not bad and all(o.ready for o in seen):
                for document_id in self._stale_ids(row):
                    self._delete(key, token, document_id)
                wanted_keep = set(row.get("pendingKeepHashes") or [])
                titles_old = list(row.get("claimTitles") or [])
                titles_old += [""] * (len(row.get("documentIds") or []) - len(titles_old))
                kept = [(i, h, t) for i, h, t in zip(row.get("documentIds") or [], row.get("claimHashes") or [],
                                                      titles_old, strict=False) if h in wanted_keep]
                final_ids = [i for i, _, _ in kept] + ids
                hashes = [h for _, h, _ in kept] + list(row.get("pendingClaimHashes") or [])
                titles = [t for _, _, t in kept] + list(row.get("pendingClaimTitles") or [])
                mode = row.get("pendingMode") or row.get("mode") or "whole"
                resplit = mode == "whole" and len(row.get("documentIds") or []) > 1
                if not self._fenced(SWAP, key, token, ids=final_ids, hashes=hashes, titles=titles,
                                    sectionHash=row.get("pendingSectionHash"), mode=mode,
                                    needsResplit=True if resplit else None, heading=heading, ord=row.get("ord")):
                    raise LeaseLost("lease lost during the swap")
                return out("synced", f"{len(ids)} submitted document(s) now searchable")
            if bad or capped:
                for document_id in ids:
                    self._delete(key, token, document_id)
                if not self._fenced(GIVE_UP, key, token):
                    raise LeaseLost("lease lost giving up")
                return out("failed", "a submitted document failed or the cap passed; the old claim stays "
                           "mapped and flagged stale; the next kb-sync retries")
            return out("pending", "embeds not landed yet")
        except LeaseLost as exc:
            return out("pending", str(exc))
        except (StoreError, KbHttpError) as exc:
            return out("failed", str(exc))
        finally:
            self._release(key, token)

    @staticmethod
    def _stale_ids(row: dict) -> list[str]:
        keep = set(row.get("pendingKeepHashes") or [])
        return [i for i, h in zip(row.get("documentIds") or [], row.get("claimHashes") or [], strict=False)
                if h not in keep]

    # -- reconcile (FR-3 for the KB) -------------------------------------------------------------
    def reconcile(self, docs: list[KbDoc]) -> list[KbOutcome]:
        """Check every mapped id against falkor-chat: missing or hash-mismatched -> re-ingest (under the normal
        flow); `currentVersion = false` -> reported, never re-ingested; unmapped documents whose hash equals a
        claim's -> adopted without re-embedding; `kb-mirror` documents no section maps -> reported as orphans."""
        results: list[KbOutcome] = list(self.finish(once=True))
        rows, _ = self.read_all()
        ctx = Context.build(docs, rows)
        by_key = {r["key"]: r for r in rows}
        for doc in docs:
            for section in doc.sections:
                if not section.claims:
                    continue
                row = by_key.get(section_key(doc.path, section))
                if not row or not row.get("documentIds"):
                    results.append(self._adopt(doc.path, section, ctx))
                elif row.get("state") == "synced" and row.get("sectionHash") == section.section_hash:
                    results.extend(self._verify(doc.path, section, row, ctx))
        rows, _ = self.read_all()
        mapped = {i for r in rows for i in [*(r.get("documentIds") or []), *(r.get("pendingDocumentIds") or [])]}
        for document_id, _hash, _current, title in self.observer.produced_by(PRODUCER):
            if document_id not in mapped:
                results.append(KbOutcome("", title or document_id, "orphan", f"kb-mirror document {document_id} "
                                         "is mapped by no section"))
        return results

    def _adopt(self, path: str, section: Section, ctx: Context) -> KbOutcome:
        out = lambda status, detail="": KbOutcome(path, section.heading, status, detail)  # noqa: E731
        claims = section.claims
        found = self.observer.current_by_hash(
            [h for c in claims for h in (c.claim_hash, _stripped_hash(c)) if h])
        picked: list[tuple[str, str, str]] = []  # (id, stored hash, title)
        exact = True
        for c in claims:
            h = c.claim_hash
            if h not in found:
                h = _stripped_hash(c) or ""
                exact = False
            if h not in found or found[h] in ctx.mapped_ids:
                return out("unadopted", f"{len(picked)} of {len(claims)} claims have a stored document")
            picked.append((found[h], h, claim_title(path, section, c)))
        key = section_key(path, section)
        token = self._acquire(key, path, section.heading)
        if token is None:
            return out("pending", "section lease is held by another actor; retry in 30 s")
        try:
            ok = self._fenced(SWAP, key, token, ids=[p[0] for p in picked], hashes=[p[1] for p in picked],
                              titles=[p[2] for p in picked], sectionHash=section.section_hash if exact else None,
                              mode=section.mode, needsResplit=None, heading=section.heading, ord=section.ord)
        except StoreError as exc:
            return out("failed", str(exc))
        finally:
            self._release(key, token)
        if not ok:
            return out("pending", "lease lost adopting")
        ctx.mapped_ids.update(p[0] for p in picked)
        return out("adopted", f"{len(picked)} document(s) mapped without re-embedding"
                   + ("" if exact else "; marker-stripped text: the next kb-sync re-ingests it"))

    def _verify(self, path: str, section: Section, row: dict, ctx: Context) -> list[KbOutcome]:
        ids, hashes = row.get("documentIds") or [], row.get("claimHashes") or []
        obs = self.observer.observe(ids)
        bad = frozenset(i for i, h in zip(ids, hashes, strict=False)
                        if obs.get(i) is None or not obs[i].exists or obs[i].hash != h or obs[i].failed)
        results = [KbOutcome(path, section.heading, "non-current", f"{i} is superseded (currentVersion = false): "
                             "two headings hold one claim; reported, not re-ingested")
                   for i in ids if i not in bad and obs[i].current is False]
        if bad:
            outcome = self.sync_section(path, section, ctx, force=True, bad_ids=bad)
            results.append(KbOutcome(path, section.heading, "re-ingested" if outcome.status == "synced"
                                     else outcome.status, f"{len(bad)} missing, mismatched or failed id(s): {outcome.detail}"))
        return results


# --- status (FR-5 for the KB) ---------------------------------------------------------------------

@dataclass(frozen=True)
class StatusLine:
    kind: str  # stale | pending | unmirrored | coarse | unmapped-id | duplicate | unregistered
    path: str
    heading: str
    detail: str

    def render(self) -> str:
        where = f"{self.path} § {self.heading}" if self.heading else self.path
        return f"{self.kind}\t{where}\t{self.detail}"


@dataclass
class StatusReport:
    last_sync_ms: int | None
    lines: list[StatusLine] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return all(line.kind == "coarse" for line in self.lines)

    def header(self) -> str:
        when = "never" if self.last_sync_ms is None else \
            datetime.fromtimestamp(self.last_sync_ms / 1000, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        return f"KB mirror: last sync {when}"


def status(docs: list[KbDoc], rows: list[dict], last_sync_ms: int | None,
           ids: Iterable[str] | None = None) -> StatusReport:
    """Flags (a) a returned id whose section's file text no longer matches its synced hash, or is pending;
    (b) any unmirrored section; (c) duplicate claims (plan 3.5). With `ids`, (a) and the coarse note cover only
    the sections holding one of them; (b), (c) and an id mapped by no section are always listed."""
    report = StatusReport(last_sync_ms)
    wanted = None if ids is None else set(ids)
    by_key = {r["key"]: r for r in rows}
    mapped_to: dict[str, str] = {}
    for r in rows:
        for i in [*(r.get("documentIds") or []), *(r.get("pendingDocumentIds") or [])]:
            mapped_to[i] = r["key"]
    live: set[str] = set()
    for doc in docs:
        for s in doc.sections:
            key = section_key(doc.path, s)
            live.add(key)
            if not s.claims:
                continue
            row = by_key.get(key)
            held = {*(row.get("documentIds") or []), *(row.get("pendingDocumentIds") or [])} if row else set()
            relevant = wanted is None or bool(held & wanted)
            read = f"read `{doc.path}` § `{s.heading}`"
            if not row or not row.get("documentIds"):
                report.lines.append(StatusLine("unmirrored", doc.path, s.heading, "no document yet: " + read))
            elif row.get("state") == "pending" or row.get("pendingDocumentIds"):
                if relevant:
                    report.lines.append(StatusLine("pending", doc.path, s.heading,
                                                   "stale — " + read + " (new version submitted, not yet searchable)"))
            elif row.get("sectionHash") != s.section_hash:
                if relevant:
                    report.lines.append(StatusLine("stale", doc.path, s.heading, "stale — " + read))
            elif row.get("needsResplit") and relevant:
                report.lines.append(StatusLine("coarse", doc.path, s.heading,
                                               "coarse: one document for a judgment split; resplit at the next pass"))
    for where in duplicate_claims(docs).values():
        report.lines.append(StatusLine("duplicate", where[0][0], where[0][1],
                                       "same claim text under " + ", ".join(f"'{hd}' ({p})" for p, hd in where)))
    for i in sorted(wanted or ()):
        if i not in mapped_to:
            report.lines.append(StatusLine("unmapped-id", "", "", f"{i} is mapped by no section"))
    for r in rows:
        if r["key"] not in live:
            report.lines.append(StatusLine("unregistered", r.get("path", ""), r.get("heading", ""),
                                           "mapped section whose heading or file is not in the registry"))
    return report

