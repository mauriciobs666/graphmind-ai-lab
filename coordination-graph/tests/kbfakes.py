"""Offline stand-ins for the KB mirror engine's three collaborators (`kbsync`): the mirror graph, falkor-chat's
REST + observation graph, and the clock. They interpret the statements by identity (the Cypher itself runs
live, `test_kb_live.py`) with the semantics the note specifies: a compare-and-set acquire, fenced writes
that match 0 rows once the token is gone, `null` removing a property, and falkor-chat's auto-supersede of a
current document with the same normalised text."""
from __future__ import annotations

import itertools

from coordgraph import kbsync
from coordgraph.kbsync import KbHttpError
from coordgraph.parse.kb import claim_hash


class Result:
    def __init__(self, rows):
        self.result_set = [list(r) for r in rows]


class Clock:
    """Shared fake time: `sleep` advances both the monotonic clock and the mirror's server clock."""

    def __init__(self, mirror: FakeMirror):
        self.t = 0.0
        self.mirror = mirror

    def sleep(self, seconds: float) -> None:
        self.t += seconds
        self.mirror.now += int(seconds * 1000)

    def monotonic(self) -> float:
        return self.t


class FakeMirror:
    def __init__(self):
        self.now = 1_000_000
        self.sections: dict[str, dict] = {}
        self.files: dict[str, dict] = {}
        self.after: dict[str, object] = {}  # statement name -> callable(params), run once after it executes
        self.fail: dict[str, Exception] = {}  # statement name -> raised once
        self.log: list[str] = []
        self.names = {id(getattr(kbsync, n)): n for n in
                      ("ACQUIRE", "RELEASE", "RENEW", "MARK", "RECORD", "CANCEL", "GIVE_UP", "SWAP", "REMOVE",
                       "READ_ONE", "READ_ALL", "READ_FILES", "UPSERT_FILE")}

    def steal(self, key: str, token: str = "other") -> None:
        """Another actor takes the lease on `key` (as after an expiry)."""
        self.sections[key]["leaseToken"] = token
        self.sections[key]["leaseUntil"] = self.now + 60_000

    @staticmethod
    def _set(props: dict, **kv) -> None:
        for k, v in kv.items():
            if v is None:
                props.pop(k, None)
            else:
                props[k] = list(v) if isinstance(v, list) else v

    def query(self, statement, params=None):
        p = params or {}
        name = self.names[id(statement)]
        self.log.append(name)
        if name in self.fail:
            raise self.fail.pop(name)
        result = getattr(self, f"_{name.lower()}")(p)
        hook = self.after.pop(name, None)
        if hook:
            hook(p)
        return result

    def _fenced(self, p):
        s = self.sections.get(p["key"])
        return s if s is not None and s.get("leaseToken") == p["token"] else None

    def _acquire(self, p):
        s = self.sections.setdefault(p["key"], {"key": p["key"], "path": p["path"], "heading": p["heading"],
                                                "state": "pending", "pendingSince": self.now})
        if s.get("leaseUntil") is None or s["leaseUntil"] < self.now:
            s["leaseToken"], s["leaseUntil"] = p["token"], self.now + p["ttlMs"]
            return Result([[s["leaseToken"], s["leaseUntil"]]])
        return Result([])

    def _release(self, p):
        s = self._fenced(p)
        if s:
            self._set(s, leaseToken=None, leaseUntil=None)
        return Result([[1 if s else 0]])

    def _renew(self, p):
        s = self._fenced(p)
        if s:
            s["leaseUntil"] = self.now + p["ttlMs"]
        return Result([[1 if s else 0]])

    _PENDING = ("pendingDocumentIds", "pendingClaimHashes", "pendingClaimTitles", "pendingKeepHashes",
                "pendingMode", "pendingSectionHash")

    def _mark(self, p):
        s = self._fenced(p)
        if s:
            self._set(s, state="pending", pendingSince=self.now, pendingSectionHash=p["sectionHash"],
                      pendingKeepHashes=p["keep"], pendingMode=p["mode"], leaseUntil=self.now + p["ttlMs"])
        return Result([[1 if s else 0]])

    def _record(self, p):
        s = self._fenced(p)
        if s:
            self._set(s, pendingDocumentIds=p["ids"], pendingClaimHashes=p["hashes"], pendingClaimTitles=p["titles"])
        return Result([[1 if s else 0]])

    def _cancel(self, p):
        s = self._fenced(p)
        if s:
            self._set(s, **dict.fromkeys(self._PENDING))
        return Result([[1 if s else 0]])

    def _give_up(self, p):
        s = self._fenced(p)
        if s:
            self._set(s, **dict.fromkeys(self._PENDING), state="synced", pendingSince=None, leaseToken=None,
                      leaseUntil=None)
        return Result([[1 if s else 0]])

    def _swap(self, p):
        s = self._fenced(p)
        if s:
            self._set(s, documentIds=p["ids"], claimHashes=p["hashes"], claimTitles=p["titles"],
                      sectionHash=p["sectionHash"], mode=p["mode"], needsResplit=p["needsResplit"],
                      heading=p["heading"], ord=p["ord"], state="synced", syncedAt=self.now,
                      **dict.fromkeys(self._PENDING), pendingSince=None, leaseToken=None, leaseUntil=None)
        return Result([[1 if s else 0]])

    def _remove(self, p):
        s = self._fenced(p)
        if s:
            del self.sections[p["key"]]
        return Result([[1 if s else 0]])

    def _read_one(self, p):
        s = self.sections.get(p["key"])
        return Result([[dict(s), self.now]] if s else [])

    def _read_all(self, p):
        return Result([[dict(s), self.now] for s in self.sections.values()])

    def _read_files(self, p):
        return Result([[k, f["sha256"], f["syncedAt"]] for k, f in self.files.items()])

    def _upsert_file(self, p):
        self.files[p["path"]] = {"sha256": p["sha256"], "syncedAt": self.now}
        return Result([[sum(1 for s in self.sections.values() if s["path"] == p["path"])]])


class FakeChat:
    """falkor-chat's `POST/DELETE /documents` and the observation reads, over one in-memory document set."""

    def __init__(self):
        self.docs: dict[str, dict] = {}
        self.calls: list[tuple] = []
        self._ids = itertools.count(1)
        self.held: set[str] = set()  # documents whose embeds have not landed
        self.failed: set[str] = set()
        self.timeout_mode: str | None = None  # 'commit' | 'late' | 'lost': client times out
        self.late: list[dict] = []
        self.supersede_foreign: str | None = None  # force a receipt naming this victim
        self.hold_new = False
        self.during_post = None  # callable run after the server commits, before the receipt returns
        self.on_delete = None  # callable(id) run at the start of a delete

    # -- plant ---------------------------------------------------------------------------------
    def plant(self, text: str, producer: str = "cobb", title: str = "planted", *, current: bool = True,
              digest: str | None = None) -> str:
        i = f"d{next(self._ids)}"
        self.docs[i] = {"id": i, "title": title, "text": text, "hash": digest or claim_hash(text),
                        "producer": producer, "current": current, "status": "ready", "chunks": 1, "embedded": 1}
        return i

    def release_embeds(self) -> None:
        self.held.clear()

    def flush_late(self) -> None:
        for doc in self.late:
            self._commit(doc)
        self.late.clear()

    def current(self, producer: str = kbsync.PRODUCER) -> list[dict]:
        return [d for d in self.docs.values() if d["producer"] == producer and d["current"]]

    # -- REST ----------------------------------------------------------------------------------
    def _commit(self, doc: dict) -> dict:
        victim = next((d for d in self.docs.values() if d["hash"] == doc["hash"] and d["current"]), None)
        self.docs[doc["id"]] = doc
        if victim:
            victim["current"] = False
        return {"documentId": doc["id"], "chunkCount": 1, "status": "processing",
                "autoSuperseded": victim is not None, "supersededDocumentId": victim["id"] if victim else None}

    def post_document(self, title: str, text: str) -> dict:
        self.calls.append(("post", title))
        i = f"d{next(self._ids)}"
        doc = {"id": i, "title": title, "text": text, "hash": claim_hash(text), "producer": kbsync.PRODUCER,
               "current": True, "status": "processing", "chunks": 1, "embedded": 0}
        if self.hold_new:
            self.held.add(i)
        if self.timeout_mode == "late":
            self.late.append(doc)
            raise KbHttpError("POST /documents: client timeout after 10s", timed_out=True)
        receipt = self._commit(doc)
        if self.timeout_mode == "commit":
            raise KbHttpError("POST /documents: client timeout after 10s", timed_out=True)
        if self.supersede_foreign:
            receipt.update(autoSuperseded=True, supersededDocumentId=self.supersede_foreign)
        if self.during_post:
            self.during_post()
        return receipt

    def delete_document(self, document_id: str) -> None:
        self.calls.append(("delete", document_id))
        if self.on_delete:
            self.on_delete(document_id)
        self.docs.pop(document_id, None)

    # -- observation ---------------------------------------------------------------------------
    def observe(self, ids):
        out = {}
        for i in ids:
            d = self.docs.get(i)
            if d is None:
                out[i] = kbsync.Obs(i, None, None, None, 0, 0)
                continue
            status = "failed" if i in self.failed else d["status"]
            embedded = 0 if i in self.held else d["chunks"]
            out[i] = kbsync.Obs(i, d["hash"], status, d["current"], d["chunks"], embedded)
        return out

    def describe(self, document_id):
        d = self.docs.get(document_id)
        return (d["hash"], d["producer"]) if d else (None, None)

    def current_by_hash(self, hashes):
        return {d["hash"]: d["id"] for d in self.docs.values() if d["current"] and d["hash"] in set(hashes)}

    def produced_by(self, agent):
        return [(d["id"], d["hash"], d["current"], d["title"]) for d in self.docs.values() if d["producer"] == agent]

    def posts(self) -> list[str]:
        return [c[1] for c in self.calls if c[0] == "post"]

    def deletes(self) -> list[str]:
        return [c[1] for c in self.calls if c[0] == "delete"]


def make_engine(**kw):
    """`(engine, mirror, chat, clock)` wired to the fakes; tokens are `t1`, `t2`, ..."""
    mirror, chat = FakeMirror(), FakeChat()
    clock = Clock(mirror)
    counter = itertools.count(1)
    engine = kbsync.KbSync(mirror, chat, chat, sleep=clock.sleep, monotonic=clock.monotonic,
                           token=lambda: f"t{next(counter)}", **{"budget_s": 2.0, "poll_s": 0.5, **kw})
    return engine, mirror, chat, clock
