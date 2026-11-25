"""CO-5 / P6-1 (e): the probe is the PRODUCT of a dev rehearsal run, scored by the same ``score_gate`` path against the committed dev gold, never typed. The writer
(``gate.write_dev_probe`` and the ``gate.py dev-probe`` verb) takes a run file (``run.json``, whose ``answers`` are the rehearsal's ``answers.jsonl`` entries), the
live snapshot and the repository, and writes the attestation the gate later checks. Offline, synthetic."""
import json

import pytest

import gate_helpers as H
from gate_helpers import G
from gate_run_helpers import YAML_EM

KE_RECORD = "cpg/eval/decisions/ccg-dev-key-errors-v1.json"
STRATA = ("http", "mcp", "scripts")


class Rehearsal:
    """A repo whose dev gold holds real (scorable) items, and the run file of a rehearsal that answered them."""

    def __init__(self, tmp_path, miss_in=None, flow_flag=False):
        self.w = H.World(qname="svc/dev.py::DevFn")
        self.repo = repo = H.make_repo(tmp_path)
        (repo / "cpg/eval/eval-manifest.yaml").write_text(YAML_EM)
        self.items = {}
        for s in STRATA:
            recs = []
            for k, iid in enumerate(H.dev_ids(s)):
                rec = {**self.w.rec, "itemId": iid, "derivedStratum": s}
                if s == miss_in and k == 0:
                    anchor = H.contract("http", "GET /nowhere")
                    rec = {**H.item(50 + k, anchor, self.w.rec["impacted"]), "itemId": iid, "derivedStratum": s}      # its anchor is not in the run's answers: a miss
                recs.append(rec)
            self.items[s] = recs
            (repo / f"cpg/eval/dev/ac5-{s}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        H.git(repo, "add", "-A")
        H.git(repo, "commit", "-qm", "dev gold")
        self.rc = H.head(repo)
        self.run = {**self.w.run(), "listKey": "d" * 64, "flowFlag": flow_flag, "withheld": {},
                    "preflight": {"recordCommit": self.rc, "harnessPin": {"sha256": {rel: "e" * 64 for rel in G.HARNESS_FILES}}}}

    def write(self, **kw):
        return G.write_dev_probe(self.repo, self.run, self.w.snap, **kw)


def confirm(repo, ids):
    (repo / KE_RECORD).write_text(json.dumps({"schema": "ccg-dev-key-errors-v1", "confirmed": [{"itemId": i, "confirmedBy": "user", "date": "2026-11-24"} for i in ids]}) + "\n")
    H.git(repo, "add", "-A")
    H.git(repo, "commit", "-qm", "confirm")


def test_a_probe_written_from_an_all_hit_rehearsal_carries_its_bindings_and_every_hit_id(tmp_path):
    r = Rehearsal(tmp_path)
    doc = r.write()
    assert doc["schema"] == G.DEV_PROBE_SCHEMA and "rehearsalId" not in doc and doc["recordCommit"] == r.rc and doc["snapshotId"] == H.SNAP
    assert doc["harnessSha256"] == r.run["preflight"]["harnessPin"]["sha256"]
    assert doc["devGoldSha256"] == {s: H.sha_of(r.repo / f"cpg/eval/dev/ac5-{s}.jsonl") for s in STRATA}
    assert {s: doc["byStratum"][s]["hitIds"] for s in STRATA} == {s: H.dev_ids(s) for s in STRATA}
    assert all(doc["byStratum"][s]["keyErrorIds"] == [] for s in STRATA)


def test_the_written_probe_is_what_the_gate_accepts_when_the_run_was_at_this_commit_and_harness(tmp_path):
    r = Rehearsal(tmp_path)
    r.run["preflight"]["harnessPin"]["sha256"] = {rel: H.sha_of(r.repo / rel) for rel in G.HARNESS_FILES if (r.repo / rel).is_file()}
    G.dev_probe_check(r.write(), r.repo, r.rc, G.STRATA_GATED)


def test_a_dev_miss_is_left_out_of_the_hit_ids_so_the_gate_refuses_the_file(tmp_path):
    r = Rehearsal(tmp_path, miss_in="scripts")
    r.run["preflight"]["harnessPin"]["sha256"] = {rel: H.sha_of(r.repo / rel) for rel in G.HARNESS_FILES if (r.repo / rel).is_file()}
    doc = r.write()
    assert doc["byStratum"]["scripts"]["hitIds"] == H.dev_ids("scripts")[1:] and doc["byStratum"]["scripts"]["keyErrorIds"] == []
    with pytest.raises(G.PrecheckRefusal, match="scripts"):
        G.dev_probe_check(doc, r.repo, r.rc, G.STRATA_GATED)


def test_only_a_non_hit_that_the_committed_record_confirms_is_written_as_a_key_error(tmp_path):
    r = Rehearsal(tmp_path, miss_in="scripts")
    missed = H.dev_ids("scripts")[0]
    confirm(r.repo, [missed, H.dev_ids("http")[0]])                                            # the second is a hit: confirming it does not make it a key-error
    r.run["preflight"]["recordCommit"] = H.head(r.repo)
    doc = r.write()
    assert doc["byStratum"]["scripts"]["keyErrorIds"] == [missed] and doc["byStratum"]["scripts"]["hitIds"] == H.dev_ids("scripts")[1:]
    assert doc["byStratum"]["http"]["keyErrorIds"] == []


def test_an_unconfirmed_miss_is_never_written_as_a_key_error(tmp_path):
    r = Rehearsal(tmp_path, miss_in="mcp")
    doc = r.write()
    assert H.dev_ids("mcp")[0] not in doc["byStratum"]["mcp"]["keyErrorIds"] + doc["byStratum"]["mcp"]["hitIds"]


def test_the_writer_refuses_a_run_of_another_snapshot_than_the_one_it_is_given(tmp_path):
    r = Rehearsal(tmp_path)
    r.run["snapshotId"] = "9" * 32
    with pytest.raises(G.Refusal, match="snapshot"):
        r.write()


def test_the_writer_refuses_a_run_that_carries_no_preflight_block_or_list_key(tmp_path):
    r = Rehearsal(tmp_path)
    del r.run["preflight"]
    with pytest.raises(G.Refusal, match="run"):
        r.write()
    r = Rehearsal(tmp_path / "b")
    del r.run["listKey"]
    with pytest.raises(G.Refusal, match="run"):
        r.write()


def test_a_run_made_while_the_flow_flag_was_true_covers_the_flow_stratum_too(tmp_path):
    r = Rehearsal(tmp_path, flow_flag=True)
    flow = [{**r.w.rec, "itemId": i, "derivedStratum": "flow"} for i in H.dev_ids("flow")]
    (r.repo / "cpg/eval/dev/ac5-flow.jsonl").write_text("".join(json.dumps(x) + "\n" for x in flow))
    H.git(r.repo, "add", "-A")
    H.git(r.repo, "commit", "-qm", "flow gold")
    doc = r.write()
    assert set(doc["byStratum"]) == {"http", "mcp", "scripts", "flow"} and doc["byStratum"]["flow"]["hitIds"] == H.dev_ids("flow")
    assert set(r.write().get("devGoldSha256")) == {"http", "mcp", "scripts", "flow"}


def test_the_cli_verb_writes_the_file_and_prints_counts_only(tmp_path, capsys):
    r = Rehearsal(tmp_path)
    run = tmp_path / "run.json"
    run.write_text(json.dumps(r.run))
    out = tmp_path / "probe.json"
    rc = G.main(["dev-probe", "--run", str(run), "--out", str(out), "--repo", str(r.repo)], snapshot=r.w.snap, repo=r.repo)
    text = capsys.readouterr().out
    assert rc == 0 and json.loads(out.read_text())["schema"] == G.DEV_PROBE_SCHEMA
    assert json.loads(text) == {"strata": {s: {"items": 3, "hits": 3, "keyErrors": 0, "withheld": 0, "withheldByShape": {}} for s in STRATA}, "complete": True}
    r2 = Rehearsal(tmp_path / "m", miss_in="http")
    run.write_text(json.dumps(r2.run))
    assert G.main(["dev-probe", "--run", str(run), "--out", str(out), "--repo", str(r2.repo)], snapshot=r2.w.snap, repo=r2.repo) == 1
    assert json.loads(capsys.readouterr().out)["complete"] is False


def test_the_cli_verb_counts_a_withheld_miss_as_complete_and_prints_the_shape_count(tmp_path, capsys, monkeypatch):
    r = Rehearsal(tmp_path, miss_in="scripts")
    miss = H.dev_ids("scripts")[0]
    monkeypatch.setattr(G, "item_withheld_shape", lambda rec, reader: "B" if rec["itemId"] == miss else None)
    run = tmp_path / "run.json"
    run.write_text(json.dumps(r.run))
    out = tmp_path / "probe.json"
    assert G.main(["dev-probe", "--run", str(run), "--out", str(out), "--repo", str(r.repo)], snapshot=r.w.snap, repo=r.repo) == 0
    row = json.loads(capsys.readouterr().out)["strata"]["scripts"]
    assert row["items"] == 3 and row["hits"] == 2 and row["keyErrors"] == 0 and row["withheld"] == 1 and row["withheldByShape"] == {"B": 1}


def test_the_writer_reads_the_record_at_the_runs_record_commit_so_a_later_widening_confirms_nothing(tmp_path):
    r = Rehearsal(tmp_path, miss_in="scripts")
    confirm(r.repo, [H.dev_ids("scripts")[0]])                          # committed AFTER r.rc, the commit the rehearsal's run recorded
    with pytest.raises(G.PrecheckRefusal, match="confirmation"):
        r.write()
