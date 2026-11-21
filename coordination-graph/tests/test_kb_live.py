"""Live tests of the KB mirror (plan S6): the real Cypher on an EPHEMERAL FalkorDB and a SCRATCH falkor-chat
(its own workspace `coordtest`, a fake embedder, a throwaway `ws:coordtest` graph). Never the shared instance,
never `falkordb-dev`, never `ws:agent-team`. Run with `-m live`.

Contract test (R10): the observation reads of `kbsync` find, in a graph falkor-chat itself wrote, the labels,
relationship types and properties they name; and falkor-chat's normaliser equals ours.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
import uuid

import pytest

from coordgraph import kbsync, store
from coordgraph.kbsync import Context, KbHttp, Observer
from coordgraph.parse.kb import claim_hash, parse_kb
from tests.fixtures.live import (  # noqa: F401  (session fixtures)
    FALKOR_CHAT,
    REPO_ROOT,
    SHARED_PORTS,
    fake_embedder,
    falkordb_container,
    scratch_falkor_chat,
)

pytestmark = pytest.mark.live


@pytest.fixture
def live(falkordb_container, scratch_falkor_chat, fake_embedder):  # noqa: F811
    c = falkordb_container
    assert c.host == "127.0.0.1" and c.port not in SHARED_PORTS and c.name.startswith("coordgraph-test-")
    assert scratch_falkor_chat.db is c and scratch_falkor_chat.workspace == "coordtest"
    fake_embedder.set_mode("ok")
    key = f"kbt-{uuid.uuid4().hex[:10]}"
    db, mirror = store.connect(c.host, c.port, key, env={store.EPHEMERAL_ENV: str(c.port)})
    store.bootstrap(db, mirror, key)
    http = KbHttp(scratch_falkor_chat.base_url, 10.0)
    observer = Observer(db.select_graph("ws:coordtest"))

    def engine(**kw):
        return kbsync.KbSync(mirror, http, observer, **{"budget_s": 15.0, "poll_s": 0.2, **kw})

    return type("Live", (), {"db": db, "mirror": mirror, "http": http, "observer": observer,
                             "engine": staticmethod(engine), "fc": scratch_falkor_chat, "fake": fake_embedder,
                             "key": key})


def run_sync(eng, doc):
    rows, _ = eng.read_all()
    return {o.heading: o for o in eng.sync_file(doc, Context.build([doc], rows), rows)}


def kb(text, path=None):
    return parse_kb(path or f"live/{uuid.uuid4().hex[:6]}.md", text.encode())


def search(live, q):
    url = f"{live.fc.base_url}/documents/search?" + urllib.parse.urlencode({"q": q, "limit": 20})
    with urllib.request.urlopen(url, timeout=15) as r:
        return {h.get("documentId") for h in json.loads(r.read())}


def wait_for(pred, what, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return
        time.sleep(0.25)
    raise TimeoutError(what)


def row_of(live, key):
    return dict(live.mirror.query(kbsync.READ_ONE, {"key": key}).result_set[0][0])


# --- R10: the contract --------------------------------------------------------------------------------

def test_contract_the_observation_reads_find_the_schema_falkor_chat_wrote(live):
    text = f"Contract claim {uuid.uuid4().hex}."
    receipt = live.http.post_document("contract", text)
    doc_id = receipt["documentId"]
    wait_for(lambda: live.observer.observe([doc_id])[doc_id].ready, "embed landed")
    o = live.observer.observe([doc_id])[doc_id]
    assert (o.hash, o.current, o.exists) == (claim_hash(text), True, True) and o.chunks >= 1 and o.embedded == o.chunks
    g = live.db.select_graph("ws:coordtest")
    labels = {r[0] for r in g.ro_query("CALL db.labels()").result_set}
    rels = {r[0] for r in g.ro_query("CALL db.relationshipTypes()").result_set}
    props = {r[0] for r in g.ro_query("CALL db.propertyKeys()").result_set}
    assert {"Document", "Chunk", "Agent"} <= labels and {"HAS_CHUNK", "INGESTED_BY"} <= rels
    assert {"documentId", "textNormalizedHash", "currentVersion", "embedding", "status", "agentId"} <= props
    assert live.observer.describe(doc_id) == (claim_hash(text), "kb-mirror")  # the seeded producer identity
    assert [r for r in live.observer.produced_by("kb-mirror") if r[0] == doc_id][0][2] is True
    assert live.observer.current_by_hash([claim_hash(text)]) == {claim_hash(text): doc_id}


def test_contract_falkor_chats_normaliser_is_ours(live):
    sys.path.insert(0, str(FALKOR_CHAT / "server"))
    try:
        from falkorchat import update_detection
    finally:
        sys.path.remove(str(FALKOR_CHAT / "server"))
    for sample in ("  Foo \n\tBAR  ", "Straße", "- a list\n  - nested", "İstanbul  ", "x"):
        assert update_detection.content_hash(update_detection.normalize_text(sample)) == claim_hash(sample)


def test_the_scratch_roster_carries_kb_mirror_from_the_seed_script(live):
    g = live.db.select_graph("ws:coordtest")
    assert g.ro_query("MATCH (a:Agent {agentId: 'kb-mirror'}) RETURN count(a)").result_set[0][0] == 1
    seed = (REPO_ROOT / "falkor-chat" / "scripts" / "seed_agent_team.sh").read_text()
    assert "kb-mirror" in seed


# --- the real statements -----------------------------------------------------------------------------

def test_the_lease_statements_are_a_compare_and_set_with_token_fencing(live):
    eng = live.engine(ttl_ms=300)
    k = f"live/k.md#{uuid.uuid4().hex[:6]}"
    a = eng._acquire(k, "live/k.md", "H")
    assert a and eng._acquire(k, "live/k.md", "H") is None  # held: 0 rows
    assert eng._fenced(kbsync.RENEW, k, a, ttlMs=300)
    assert not eng._fenced(kbsync.RENEW, k, "wrong")
    time.sleep(0.5)  # expired
    b = eng._acquire(k, "live/k.md", "H")
    assert b and b != a
    assert not eng._fenced(kbsync.RENEW, k, a, ttlMs=300) and not eng._fenced(kbsync.RELEASE, k, a)  # stale holder
    assert not eng._fenced(kbsync.MARK, k, a, sectionHash="x", mode="whole", keep=[], ttlMs=300)
    assert eng._fenced(kbsync.RELEASE, k, b)
    props = row_of(live, k)
    assert "leaseToken" not in props and "leaseUntil" not in props  # `null` removes the property


# --- end to end ------------------------------------------------------------------------------------------

def test_ac8_a_section_is_searchable_after_sync_gone_after_removal_and_flagged_when_edited_behind_the_hook(live):
    claim = f"Zebra{uuid.uuid4().hex[:8]} lives on the savannah."
    eng = live.engine()
    doc = kb(f"# T\n\n## Zebra fact\n\n{claim}\n")
    out = run_sync(eng, doc)
    assert out["Zebra fact"].status == "synced"
    key = kbsync.section_key(doc.path, doc.sections[0])
    [doc_id] = row_of(live, key)["documentIds"]
    assert doc_id in search(live, claim)
    rows, _ = eng.read_all()
    edited = parse_kb(doc.path, f"# T\n\n## Zebra fact\n\n{claim} Edited.\n".encode())
    report = kbsync.status([edited], rows, 1, ids=[doc_id])
    assert [line.kind for line in report.lines] == ["stale"]  # hook bypassed: flagged
    gone = parse_kb(doc.path, b"# T\n\nnothing\n")
    rows, _ = eng.read_all()
    out = {o.heading: o for o in eng.sync_file(gone, Context.build([gone], rows), rows)}
    assert out["Zebra fact"].status == "removed"
    assert doc_id not in search(live, claim)
    assert live.observer.observe([doc_id])[doc_id].exists is False
    assert not live.mirror.query(kbsync.READ_ONE, {"key": key}).result_set  # the record is dropped


def test_an_edit_replaces_the_claim_append_before_delete_and_the_old_one_is_gone(live):
    eng = live.engine()
    tag = uuid.uuid4().hex[:8]
    path = f"live/{tag}.md"
    doc = kb(f"# T\n\n## Fact\n\nOld text {tag}.\n", path)
    run_sync(eng, doc)
    key = kbsync.section_key(path, doc.sections[0])
    [old] = row_of(live, key)["documentIds"]
    doc2 = kb(f"# T\n\n## Fact\n\nNew text {tag}.\n", path)
    assert run_sync(eng, doc2)["Fact"].status == "synced"
    [new] = row_of(live, key)["documentIds"]
    assert new != old and live.observer.observe([old])[old].exists is False
    assert live.observer.observe([new])[new].ready
    assert doc2.sections[0].claims[0].claim_hash == row_of(live, key)["claimHashes"][0]


def test_a_slow_embedder_leaves_the_section_pending_then_finish_swaps_it(live):
    eng = live.engine(budget_s=1.0)
    tag = uuid.uuid4().hex[:8]
    path = f"live/{tag}.md"
    doc = kb(f"# T\n\n## Fact\n\nFirst {tag}.\n", path)
    run_sync(eng, doc)
    key = kbsync.section_key(path, doc.sections[0])
    [old] = row_of(live, key)["documentIds"]
    live.fake.set_mode("slow", 4)
    out = run_sync(eng, kb(f"# T\n\n## Fact\n\nSecond {tag}.\n", path))
    assert out["Fact"].status == "pending"
    r = row_of(live, key)
    assert r["state"] == "pending" and r["documentIds"] == [old] and len(r["pendingDocumentIds"]) == 1
    assert live.observer.observe([old])[old].current is True  # old claim still served
    live.fake.set_mode("ok")
    wait_for(lambda: all(o.ready for o in live.observer.observe(r["pendingDocumentIds"]).values()), "embeds land", 40)
    eng2 = live.engine()
    [done] = [o for o in eng2.finish(once=True) if o.heading == "Fact"]
    assert done.status == "synced"
    r = row_of(live, key)
    assert len(r["documentIds"]) == 1 and r["documentIds"][0] != old and "pendingDocumentIds" not in r
    assert live.observer.observe([old])[old].exists is False


def test_an_embedder_killed_mid_sync_never_loses_the_old_claim(live):
    eng = live.engine(budget_s=2.0)
    tag = uuid.uuid4().hex[:8]
    path = f"live/{tag}.md"
    doc = kb(f"# T\n\n## Fact\n\nKept {tag}.\n", path)
    run_sync(eng, doc)
    key = kbsync.section_key(path, doc.sections[0])
    [old] = row_of(live, key)["documentIds"]
    live.fake.set_mode("down")
    try:
        out = run_sync(eng, kb(f"# T\n\n## Fact\n\nChanged {tag}.\n", path))
    finally:
        live.fake.set_mode("ok")
    assert out["Fact"].status in ("pending", "failed")
    assert live.observer.observe([old])[old].current is True and row_of(live, key)["documentIds"] == [old]


def test_m8_a_duplicate_heading_is_refused_and_reconcile_reports_without_reingesting_a_non_current_id(live):
    eng = live.engine()
    tag = uuid.uuid4().hex[:8]
    path = f"live/{tag}.md"
    doc = kb(f"# T\n\n## First\n\nSame claim {tag}.\n\n## Second\n\nsame   CLAIM {tag}.\n", path)
    out = run_sync(eng, doc)
    assert out["First"].status == "synced" and out["Second"].status == "refused" and "'First'" in out["Second"].detail
    key = kbsync.section_key(path, doc.sections[0])
    [first] = row_of(live, key)["documentIds"]
    assert live.observer.observe([first])[first].current is True
    live.http.post_document("twin", f"Same claim {tag}.")  # supersedes `first` behind the mapping's back
    assert live.observer.observe([first])[first].current is False
    only_first = kb(f"# T\n\n## First\n\nSame claim {tag}.\n", path)
    before = {r[0] for r in live.observer.produced_by("kb-mirror")}
    results = [o for o in eng.reconcile([only_first]) if o.heading == "First"]
    assert [o.status for o in results] == ["non-current"]
    assert {r[0] for r in live.observer.produced_by("kb-mirror")} == before  # nothing re-ingested


def test_the_17_marker_stripped_claims_of_falkordb_reference_are_reingested_by_the_first_sync(live):
    """The S6 row's count, observed: the stored stripped claims are adopted by reconcile, then the first sync
    re-ingests every one under append-before-delete."""
    path = "claude/graph-dba/falkordb-reference.md"
    real = parse_kb(path, (REPO_ROOT / path).read_bytes())
    real.sections = [s for s in real.sections if s.mode == "bullets"]  # the whole sections are not the subject
    stripped = [kbsync._BULLET.sub("", c.text, count=1) + f" [{live.key}]" for s in real.sections for c in s.claims]
    # tag each stored claim and the file's claims alike, so this run's documents are its own
    for s in real.sections:
        for i, c in enumerate(s.claims):
            tagged = type(c)(c.text + f" [{live.key}]", c.title, claim_hash(c.text + f" [{live.key}]"), c.key)
            s.claims[i] = tagged
    old = [live.http.post_document("falkordb-reference — stored", t)["documentId"] for t in stripped]
    wait_for(lambda: all(o.ready for o in live.observer.observe(old).values()), "stored claims embedded", 60)
    assert len(old) == 17
    eng = live.engine(budget_s=60.0)
    # the stored claims were ingested by kb-mirror here; planted claims adopted by hash, marker stripped
    adopted = [o for o in eng.reconcile([real]) if o.status == "adopted"]
    assert len(adopted) == 3
    posts_before = {r[0] for r in live.observer.produced_by("kb-mirror")}
    rows, _ = eng.read_all()
    outcomes = eng.sync_file(real, Context.build([real], rows), rows)
    assert {o.status for o in outcomes} == {"synced"}
    after = {r[0]: r for r in live.observer.produced_by("kb-mirror")}
    created = set(after) - posts_before
    assert len(created) == 17 and not (set(old) & set(after))  # 17 new, all 17 stored ones deleted
    assert all(after[i][3].startswith("falkordb-reference — ") for i in created)
    texts = [r[0] for r in live.observer._rows(
        "UNWIND $ids AS i MATCH (d:Document {documentId: i}) RETURN d.text", {"ids": sorted(created)})]
    assert len(texts) == 17 and all(t[0] in "-*0123456789" for t in texts)  # the list marker is kept


def _fail_document(live, document_id, *, embedded):
    """Reproduce a failed extract job on the scratch graph (SERVER.md 1): `status = failed`; an unembedded one
    also loses its chunk vectors (the embed job failed too)."""
    g = live.db.select_graph("ws:coordtest")
    g.query("MATCH (d:Document {documentId: $id}) SET d.status = 'failed'", {"id": document_id})
    if not embedded:
        g.query("MATCH (:Document {documentId: $id})-[:HAS_CHUNK]->(c:Chunk) SET c.embedding = null",
                {"id": document_id})


def _synced_fact(live, tag, word):
    eng = live.engine()
    path = f"live/{tag}.md"
    doc = kb(f"# T\n\n## Fact\n\nRetry {tag} {word}.\n", path)
    run_sync(eng, doc)
    key = kbsync.section_key(path, doc.sections[0])
    [old] = row_of(live, key)["documentIds"]
    return eng, doc, key, old


def test_a_failed_but_embedded_document_is_degraded_and_stays_current_and_searchable(live):
    eng, doc, key, old = _synced_fact(live, uuid.uuid4().hex[:8], "marmot")
    _fail_document(live, old, embedded=True)
    out = run_sync(eng, doc)
    assert out["Fact"].status == "degraded", out["Fact"]
    assert row_of(live, key)["documentIds"] == [old] and live.observer.observe([old])[old].current is True
    assert old in search(live, doc.sections[0].claims[0].text)


def test_an_unsearchable_failed_document_is_resubmitted_counted_and_the_count_cleared_on_success(live):
    eng, doc, key, old = _synced_fact(live, uuid.uuid4().hex[:8], "otter")
    _fail_document(live, old, embedded=False)
    live.fake.set_mode("down")
    try:
        out = run_sync(live.engine(budget_s=2.0), doc)
    finally:
        live.fake.set_mode("ok")
    assert out["Fact"].status in ("pending", "failed")
    assert row_of(live, key)["retryCount"] == 1 and row_of(live, key).get("lastRetryAt")
    eng.finish()  # settles or gives up the in-flight attempt
    row = row_of(live, key)
    if row["documentIds"] == [old]:  # gave up: the second attempt waits out the backoff, so lift it for the test
        live.mirror.query("MATCH (s:KbSection {key: $k}) SET s.lastRetryAt = 0", {"k": key})
        assert run_sync(eng, doc)["Fact"].status == "synced"
    [new] = row_of(live, key)["documentIds"]
    assert new != old and live.observer.observe([new])[new].ready and live.observer.observe([old])[old].exists is False
    assert "retryCount" not in row_of(live, key)
    assert run_sync(eng, doc)["Fact"].status == "unchanged"
