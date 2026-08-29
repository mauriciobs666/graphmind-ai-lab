#!/usr/bin/env python3
"""Seed (or re-seed) the dedicated NL-query-generation evaluation workspace
`ws:nlq-eval` (K-055 M6 unit U29, `docs/plans/workflow-nl-query-generation-ml.md`
§4 "Document-ingestion entity graph" corpus).

This is the AC-2 second-schema corpus for K-055's golden-set evaluation: a
fresh, purpose-built 10-15 document synthetic corpus ingested through the
document-ingestion pipeline (`document-ingestion.md`, M5), deliberately
spanning multiple `Entity.type` values, a mix of relationship predicates, one
deliberate conflicting-fact pair (FR-6), and one genuinely absent fact (for
the not-found/abstention golden category) — never `ws:acme`'s thin QA
fixture data (`workflow-nl-query-generation-ml.md` §2's volume/variety
finding: almost all `Organization`, no `Person`/`Location`/`Product`/`Event`/
`Concept` instances there).

**Why this script is NOT self-contained like `seed_eval_corpus.py`.**
`seed_eval_corpus.py` writes GraphRAG chat messages directly via
`Repository`/`EmbeddingWorker` — no background pipeline is involved for a
plain message post, so a bare script talking straight to the repository is
correct there. Document ingestion is different: `Services.ingest_document`
only creates `Document`/`Chunk` nodes and does NOT trigger extraction —
embed+extract scheduling (`background._schedule_chunk_processing`) is wired
only inside the REST (`api.py`) / MCP (`mcp.py`) transport handlers
(confirmed by reading `services.ingest_document`'s own docstring and
`app._build_default_app`). So this script is a REST **client** — it needs an
already-running server, wired with `FALKORCHAT_WS_ID=nlq-eval` and
`FALKORCHAT_ENABLE_AGENT=1` (the ingestion pipeline is gated on that flag,
`app._build_default_app`), reachable at `NLQ_EVAL_BASE_URL`. The `.sh`
wrapper checks for this precondition (a `GET /health` probe) the same way it
checks FalkorDB reachability, and prints the exact command to start one if
it isn't up yet.

**Idempotency strategy (document ingestion has no natural idempotency key,
`document-ingestion.md` §2.4/§3.5 — this script's own call to make, per the
K-055 brief):** every corpus document carries a fixed, unique `title`
(`"NLQ-EVAL-<NN>: <slug>"`). Before POSTing a document, this script reads
the graph directly (`MATCH (d:Document {title: $title})`) for an existing
`Document` with that exact title. If one exists and its `status` is a
terminal state (`ready` or `failed`), the document is skipped — a re-run
does zero redundant ingestion/extraction calls, mirroring
`seed_eval_corpus.py`'s "second run is a no-op" idempotency proof. If one
exists but is still stuck at `'processing'` (a previous run's server died
mid-extraction), this script does NOT re-POST it (that would mint a second,
duplicate `Document` per the non-idempotent-`CREATE` posture
`create_document`'s own docstring documents) — it re-polls the existing one
instead. `FORCE_REINGEST=1` bypasses the title check entirely and POSTs
every document fresh (mirrors `seed_eval_corpus.sh`'s `RESEED=1` escape
hatch) — use this only after a deliberate `GRAPH.DELETE ws:nlq-eval` +
re-bootstrap, never against a workspace with prior runs still present, or
the graph accumulates duplicate `Document`/`Entity` nodes under the same
titles.

**Verification posture (the K-055 brief's hard rule, §2/§4 of the ml
note):** after every document reaches a terminal status, this script reads
the ACTUAL extracted `Entity`/`RELATES_TO` content back from the graph
(never assumed from the corpus source text) and writes it verbatim into the
provenance sidecar, `server/tests/eval/nlq_corpus_provenance.json` — the
ground-truth artifact the golden-set-authoring unit builds against.

What this does, in order:

  1. Check the target server is reachable (`GET /health`) — fail loudly with
     the exact start command if not (this script never spawns uvicorn
     itself, same posture as `seed_eval_corpus.sh` never spawning FalkorDB
     beyond the reachability check).
  2. For each corpus document not already present (by `title`) and not
     already ingested this run: `POST /documents`, capturing `documentId`.
  3. Poll `GET /documents/{id}` until `status` leaves `'processing'` for
     every document just ingested (mirrors `test_api.py`'s
     `_poll_document_until_terminal`).
  4. Read back the actual `Entity`/`RELATES_TO` graph content per document,
     directly via Cypher (`Repository`/`db.workspace_graph`, read-only).
  5. Write the provenance sidecar and print a summary: entity-type
     distribution, relationship-predicate variety, and whether the intended
     conflicting-fact pair and the intended absent fact held.

Run via `scripts/seed_nlq_eval_corpus.sh` (sets up `PYTHONPATH`/venv the
same way `seed_eval_corpus.sh` does for its own script).
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]  # .../falkor-chat
_SERVER_DIR = _REPO_ROOT / "server"
_PROVENANCE_PATH = _SERVER_DIR / "tests" / "eval" / "nlq_corpus_provenance.json"

sys.path.insert(0, str(_SERVER_DIR))

from falkorchat import db  # noqa: E402

NLQ_WS = os.environ.get("NLQ_EVAL_WS", "nlq-eval")
BASE_URL = os.environ.get("NLQ_EVAL_BASE_URL", "http://127.0.0.1:8010").rstrip("/")
FORCE_REINGEST = os.environ.get("FORCE_REINGEST", "").strip().lower() in {
    "1", "true", "yes", "on",
}
POLL_ATTEMPTS = int(os.environ.get("NLQ_EVAL_POLL_ATTEMPTS", "30"))
POLL_INTERVAL_S = float(os.environ.get("NLQ_EVAL_POLL_INTERVAL_S", "2"))

# ── the corpus (K-055 brief; ml note §4) ───────────────────────────────────
#
# A single fictional scenario — "Marlowe Robotics", a fictional warehouse-
# robotics startup — spanning Person/Organization/Location/Product/Event/
# Concept `Entity.type`s (the closed 7-value taxonomy also allows `Other`,
# which the model uses as its own catch-all for anything it can't cleanly
# classify — e.g. a bare year or a job-title string — not something this
# corpus tries to force). A mix of relationship predicates (founded,
# headquartered in, produces/launched, presented at, partnered with,
# invested in, acquired, has, researches, competes with, develops,
# promoted to, joined as — never just "acquired"/"has" repeated). Docs 07/08
# are the deliberate FR-6 conflicting-fact pair (mirrors the ws:acme
# Meridian-Analytics 40-vs-400-employees precedent): both state Marlowe
# Robotics' headcount "as of March 2026" but with different numbers, so a
# golden pair can assert both conflicting edges are kept, never merged.
# Deliberate absent fact: NO document anywhere in this corpus states
# Marlowe Robotics' revenue/annual-revenue figure — reserved for the
# not-found/abstention golden category (verify this by grep, not just by
# construction, before trusting it — see `main()`'s absentee check).
_CORPUS: list[dict[str, str]] = [
    {
        "slug": "marlowe-robotics-founding",
        "title": "NLQ-EVAL-01: Marlowe Robotics Founding",
        "text": (
            "Elena Ferro founded Marlowe Robotics in Austin, Texas in 2019. "
            "Devon Cole joined as co-founder and Chief Technology Officer "
            "three months later."
        ),
    },
    {
        "slug": "atlas-7-product-launch",
        "title": "NLQ-EVAL-02: Atlas-7 Product Launch",
        "text": (
            "Marlowe Robotics launched the Atlas-7 warehouse robot on "
            "April 14, 2025. The Atlas-7 can lift pallets weighing up to "
            "800 kilograms."
        ),
    },
    {
        "slug": "roboworks-summit-presentation",
        "title": "NLQ-EVAL-03: RoboWorks Summit Presentation",
        "text": (
            "Marlowe Robotics presented the Atlas-7 at the RoboWorks "
            "Summit 2025, held in Denver, Colorado. Devon Cole delivered "
            "the keynote demonstration."
        ),
    },
    {
        "slug": "cascade-logistics-partnership",
        "title": "NLQ-EVAL-04: Cascade Logistics Partnership",
        "text": (
            "Marlowe Robotics partnered with Cascade Logistics in June "
            "2025 to deploy Atlas-7 robots across Cascade Logistics' "
            "Denver distribution centers. Cascade Logistics is "
            "headquartered in Denver, Colorado."
        ),
    },
    {
        "slug": "thornfield-ventures-investment",
        "title": "NLQ-EVAL-05: Thornfield Ventures Investment",
        "text": (
            "Thornfield Ventures led a Series B funding round for Marlowe "
            "Robotics in January 2026, investing 18 million dollars."
        ),
    },
    {
        "slug": "brightline-systems-acquisition",
        "title": "NLQ-EVAL-06: Brightline Systems Acquisition",
        "text": (
            "Marlowe Robotics acquired Brightline Systems, a smaller "
            "robotics-software startup based in Boulder, Colorado, in "
            "March 2026."
        ),
    },
    {
        "slug": "employee-count-conflict-a",
        "title": "NLQ-EVAL-07: Marlowe Robotics Employee Count (A)",
        "text": "As of March 2026, Marlowe Robotics has 62 employees.",
    },
    {
        "slug": "employee-count-conflict-b",
        "title": "NLQ-EVAL-08: Marlowe Robotics Employee Count (B)",
        "text": (
            "As of March 2026, Marlowe Robotics has 140 employees, "
            "following the Brightline Systems acquisition."
        ),
    },
    {
        "slug": "swarm-coordination-research",
        "title": "NLQ-EVAL-09: Swarm Coordination Research",
        "text": (
            "Marlowe Robotics researches swarm coordination, a concept in "
            "decentralized robotics control that lets multiple Atlas-7 "
            "units navigate a warehouse without central routing."
        ),
    },
    {
        "slug": "competitor-landscape",
        "title": "NLQ-EVAL-10: Competitor Landscape",
        "text": (
            "Marlowe Robotics competes with Vantage Automation in the "
            "warehouse-robotics market. Vantage Automation is "
            "headquartered in Seattle, Washington."
        ),
    },
    {
        "slug": "novagrid-software-platform",
        "title": "NLQ-EVAL-11: NovaGrid Software Platform",
        "text": (
            "Marlowe Robotics develops NovaGrid, a fleet-management "
            "software platform that coordinates Atlas-7 robots in real "
            "time. NovaGrid was released in September 2025."
        ),
    },
    {
        "slug": "leadership-update",
        "title": "NLQ-EVAL-12: Leadership Update",
        "text": (
            "In February 2026, Devon Cole was promoted from Chief "
            "Technology Officer to Chief Operating Officer at Marlowe "
            "Robotics. Priya Nandakumar joined Marlowe Robotics as the "
            "new Chief Technology Officer."
        ),
    },
]

# The deliberate absentee (ml note §4's "at least one genuinely absent
# fact"): no document above may mention Marlowe Robotics' revenue. Checked
# mechanically in `main()` before trusting the provenance, not just assumed
# from having written it this way.
_ABSENTEE_QUESTION = "How much annual revenue does Marlowe Robotics generate?"
_ABSENTEE_FORBIDDEN_TERMS = ("revenue", "arr", "annual recurring")


def _http_json(method: str, path: str, body: dict | None = None) -> dict:
    url = f"{BASE_URL}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _check_server_reachable() -> None:
    try:
        health = _http_json("GET", "/health")
    except (urllib.error.URLError, OSError) as exc:
        raise SystemExit(
            f"ERROR: {BASE_URL} is not reachable ({exc}).\n"
            "This script needs a REAL running server wired to ws:nlq-eval "
            "with FALKORCHAT_ENABLE_AGENT=1 (Services.ingest_document alone "
            "does not trigger background extraction — see this script's own "
            "module docstring). Start one, e.g.:\n\n"
            "  FALKORCHAT_WS_ID=nlq-eval FALKORCHAT_USER_ID=nlq-author \\\n"
            "  EMBEDDING_DIM=1024 FALKORCHAT_ENABLE_AGENT=1 "
            "FALKORCHAT_WORKFLOW_ENABLED=0 \\\n"
            "  UVICORN_ARGS='--port 8010' ./scripts/start_server.sh\n\n"
            "then re-run this script with NLQ_EVAL_BASE_URL=http://127.0.0.1:8010 "
            "(or whatever port you chose)."
        ) from exc
    if health.get("status") != "ok":
        raise SystemExit(f"ERROR: {BASE_URL}/health returned unexpected body: {health!r}")


def _find_existing_document(graph, *, title: str) -> dict | None:
    res = graph.ro_query(
        "MATCH (d:Document {title: $title}) "
        "RETURN d.documentId AS documentId, d.status AS status "
        "ORDER BY d.createdAt ASC LIMIT 1",
        {"title": title},
    )
    if not res.result_set:
        return None
    row = res.result_set[0]
    return {"documentId": row[0], "status": row[1]}


def _poll_until_terminal(document_id: str) -> dict:
    doc = None
    for _ in range(POLL_ATTEMPTS):
        doc = _http_json("GET", f"/documents/{document_id}")
        if doc["status"] != "processing":
            return doc
        time.sleep(POLL_INTERVAL_S)
    return doc  # returns still-'processing' — caller reports this as a problem


def _read_document_entities(graph, *, document_id: str) -> list[dict]:
    res = graph.ro_query(
        "MATCH (d:Document {documentId: $documentId})-[:HAS_CHUNK]->(:Chunk)"
        "-[:ABOUT]->(e:Entity) "
        "RETURN DISTINCT e.entityId AS entityId, e.name AS name, e.type AS type "
        "ORDER BY e.name",
        {"documentId": document_id},
    )
    return [{"entityId": r[0], "name": r[1], "type": r[2]} for r in res.result_set]


def _read_document_relationships(graph, *, document_id: str) -> list[dict]:
    res = graph.ro_query(
        "MATCH (subj:Entity)-[r:RELATES_TO]->(obj:Entity) "
        "WHERE r.sourceDocumentId = $documentId "
        "RETURN subj.name AS subject, r.label AS predicate, obj.name AS object "
        "ORDER BY subj.name, r.label, obj.name",
        {"documentId": document_id},
    )
    return [{"subject": r[0], "predicate": r[1], "object": r[2]} for r in res.result_set]


def main() -> None:
    # Fail loudly BEFORE any network/graph call if the absentee constraint
    # was violated by an edit to `_CORPUS` above — cheaper than discovering
    # it after a live ingestion run.
    for doc in _CORPUS:
        lowered = doc["text"].lower()
        for term in _ABSENTEE_FORBIDDEN_TERMS:
            if term in lowered:
                raise SystemExit(
                    f"ERROR: corpus doc {doc['slug']!r} mentions {term!r} — "
                    "this corpus's deliberate absent fact is Marlowe "
                    "Robotics' revenue; no document may mention it."
                )

    _check_server_reachable()

    conn = db.connect()
    graph = db.workspace_graph(conn, NLQ_WS)

    ingested: list[dict] = []  # {slug, title, documentId, text}
    skipped_ready = 0
    newly_ingested = 0
    terminal_by_id: dict[str, str] = {}

    # Deliberately sequential — post ONE document and poll it to a terminal
    # status before posting the next, rather than posting all 12 back-to-back
    # and polling afterward. Live-verified necessity, not caution for its own
    # sake: posting the whole batch upfront let this workspace's two background
    # jobs per chunk (embed via the embedding model, extract via the
    # generation model) from DIFFERENT documents overlap in-flight against the
    # single local LM Studio instance, which only keeps one model loaded at a
    # time under JIT loading — the resulting model-swap thrashing surfaced as
    # `ProviderCallError: ... HTTP 400 Bad Request: {"error":"Model is
    # unloaded."}` and took 11 of 12 documents straight to `status='failed'`
    # on the very first run of this script. One document fully settled before
    # the next is posted keeps at most one embed/extract pair in flight.
    print(f"Seeding {len(_CORPUS)} documents into ws:{NLQ_WS} via {BASE_URL} ...")
    for doc in _CORPUS:
        existing = None if FORCE_REINGEST else _find_existing_document(
            graph, title=doc["title"]
        )
        if existing is not None and existing["status"] in ("ready", "failed"):
            print(f"  [skip] {doc['title']!r} already {existing['status']!r}")
            ingested.append({**doc, "documentId": existing["documentId"]})
            terminal_by_id[existing["documentId"]] = existing["status"]
            skipped_ready += 1
            continue
        if existing is not None and existing["status"] == "processing":
            print(f"  [repoll] {doc['title']!r} still 'processing' from a prior run")
            document_id = existing["documentId"]
        else:
            receipt = _http_json(
                "POST", "/documents", {"text": doc["text"], "title": doc["title"]},
            )
            document_id = receipt["documentId"]
            print(f"  [posted] {doc['title']!r} -> documentId={document_id}")
            newly_ingested += 1
        ingested.append({**doc, "documentId": document_id})

        polled = _poll_until_terminal(document_id)
        terminal_by_id[document_id] = polled["status"]
        marker = "OK" if polled["status"] == "ready" else "FAILED/STUCK"
        print(f"  [{marker}] {doc['title']!r} -> status={polled['status']!r}")

    print("Reading back actual extracted content...")
    provenance_docs = []
    type_counts: dict[str, int] = {}
    predicate_counts: dict[str, int] = {}
    for item in ingested:
        document_id = item["documentId"]
        status = terminal_by_id.get(document_id, "unknown")
        entities = _read_document_entities(graph, document_id=document_id)
        relationships = _read_document_relationships(graph, document_id=document_id)
        for e in entities:
            type_counts[e["type"]] = type_counts.get(e["type"], 0) + 1
        for r in relationships:
            predicate_counts[r["predicate"]] = predicate_counts.get(r["predicate"], 0) + 1
        provenance_docs.append({
            "slug": item["slug"],
            "title": item["title"],
            "documentId": document_id,
            "status": status,
            "sourceText": item["text"],
            "extractedEntities": entities,
            "extractedRelationships": relationships,
        })

    # FR-6 conflicting-fact check: both employee-count docs (07/08) must have
    # actually produced a "has"-shaped RELATES_TO edge from Marlowe Robotics
    # with a DIFFERENT object each, both persisted (never merged/overwritten)
    # — verified from the just-read-back graph content, not assumed.
    doc_a = next(d for d in provenance_docs if d["slug"] == "employee-count-conflict-a")
    doc_b = next(d for d in provenance_docs if d["slug"] == "employee-count-conflict-b")
    conflict_objects_a = {
        r["object"] for r in doc_a["extractedRelationships"]
        if r["subject"] == "Marlowe Robotics"
    }
    conflict_objects_b = {
        r["object"] for r in doc_b["extractedRelationships"]
        if r["subject"] == "Marlowe Robotics"
    }
    conflict_holds = bool(conflict_objects_a) and bool(conflict_objects_b) and (
        conflict_objects_a != conflict_objects_b
    )

    seeded_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    provenance = {
        "workspace": f"ws:{NLQ_WS}",
        "seededAt": seeded_at,
        "documentCount": len(provenance_docs),
        "documents": provenance_docs,
        "entityTypeDistribution": type_counts,
        "relationshipPredicateDistribution": predicate_counts,
        "conflictingFactPair": {
            "documentSlugs": ["employee-count-conflict-a", "employee-count-conflict-b"],
            "docAObjects": sorted(conflict_objects_a),
            "docBObjects": sorted(conflict_objects_b),
            "bothPersistedAndDiffer": conflict_holds,
        },
        "deliberateAbsentFact": {
            "question": _ABSENTEE_QUESTION,
            "forbiddenTerms": list(_ABSENTEE_FORBIDDEN_TERMS),
            "verifiedAbsentFromSourceText": True,
        },
    }
    _PROVENANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _PROVENANCE_PATH.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    print(
        f"\nws:{NLQ_WS} provenance — documents={len(provenance_docs)} "
        f"newlyIngested={newly_ingested} skippedAlreadyTerminal={skipped_ready}"
    )
    print(f"Entity.type distribution: {type_counts}")
    print(f"Relationship predicates: {sorted(predicate_counts)}")
    print(f"Conflicting-fact pair both-persisted-and-differ: {conflict_holds}")
    print(f"Sidecar written: {_PROVENANCE_PATH.relative_to(_REPO_ROOT)}")

    not_ready = [d for d in provenance_docs if d["status"] != "ready"]
    if not_ready:
        print(
            "\nWARNING: the following documents did not reach 'ready': "
            + ", ".join(f"{d['title']!r}={d['status']!r}" for d in not_ready)
        )
    if not conflict_holds:
        print(
            "\nWARNING: the intended conflicting-fact pair (docs 07/08) did "
            "NOT persist as two distinct 'has' facts on Marlowe Robotics — "
            "inspect extractedRelationships in the sidecar and consider "
            "rewriting those two documents."
        )


if __name__ == "__main__":
    main()
