"""The campaign ledger, reconciliation, halts and `resolve` (typed-decision-models U5.6).

Rows: plan4 `docs/plans/typed-decision-models4.md` §5 U5.6, plan5 `typed-decision-models5.md` §4
U5.6, plan6 `typed-decision-models6.md` §5 U5.6, the carried rulings in
`docs/plans/typed-decision-models-coordination.md` (the commit-point event and planted cases
(a)–(e), the re-ingest seam, retired names) and impl review Pass 9's caller items.

Runs are staged from the clean fixture run (`tests/fixtures/transcripts/`, H1's foreground shape)
with its paths, opaque name and session id substituted, under a fake `claude_dir`. Nothing calls
`claude`; the clock is a test clock.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from decisionmodels import audit, campaign, exchange, ingest, seeds
from decisionmodels.campaign import Campaign, CampaignRefused, Serve, Stop
from decisionmodels.ingest import AnswerRefused

FIXTURES = Path(__file__).parent / "fixtures" / "transcripts"
F_SESSION = "5e551011-0000-4000-8000-0000000000d1"
F_AGENT = "a5c1e8d2b7f40936e"
F_OPAQUE = "q7m2x4k9p3r8t6v5"
GUARD_COMMAND = "python3 guard_hook.py"
SPAWN_TEMPLATE = (
    "Read {request_path} with offset 1, limit 1. Its `pages` field lists [offset, limit] pairs: "
    "read each remaining page with exactly that offset and limit. Follow the instructions on "
    "line 2, write the answer where the header says, then reply `done`."
)
ROLES = {
    "generator": {"agentType": "dm-generator", "model": "claude-opus-5-5", "effort": "medium",
                  "spawnTemplate": SPAWN_TEMPLATE, "instructions": []},
    "labeler": {"agentType": "dm-labeler", "model": "claude-sonnet-5-5", "effort": "high",
                "spawnTemplate": SPAWN_TEMPLATE, "instructions": []},
}
T0 = datetime(2026, 11, 2, 9, 0, tzinfo=UTC)


class Clock:
    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> datetime:
        self.now += timedelta(seconds=1)
        return self.now


def guard_item(index: int) -> dict[str, Any]:
    return {
        "condition": "the user has stated a budget for the request",
        "path": "turns",
        "understanding": {},
        "turns": [{"role": "user", "displayName": "Ana", "text": f"Orçamento de R$ {index}00"}],
    }


def write_fixture(path: Path, count: int) -> Path:
    path.write_text("".join(json.dumps(guard_item(i), ensure_ascii=False) + "\n"
                            for i in range(count)), "utf-8")
    return path


def smoke(tmp_path: Path, *, items: int = 2, clock: Clock | None = None,
          bound: timedelta = campaign.DEFAULT_LAUNCH_FAIL_BOUND, root: Path | None = None
          ) -> Campaign:
    """An initialised smoke campaign with declared expectations: one label batch `bat_0001`."""
    fixture = write_fixture(tmp_path / "fixture.jsonl", items)
    camp = Campaign(root or tmp_path / "camp", claude_dir=tmp_path / "claude",
                    clock=clock or Clock())
    camp.init(point="guard-judge", phase="smoke", fixture=fixture, launch_fail_bound=bound)
    camp.record_expectations(guard_command=GUARD_COMMAND, roles=ROLES)
    return camp


def fresh(camp: Campaign) -> Campaign:
    """The same root as the next run sees it: a new `run`, so a new instance."""
    return Campaign(camp.root, claude_dir=camp.claude_dir, clock=camp.clock)


def ledger(camp: Campaign) -> bytes:
    return camp.ledger_path.read_bytes()


def outcomes(camp: Campaign) -> list[dict[str, Any]]:
    return [e for e in camp.events() if e["event"] == "outcome"]


# --- staging a run -------------------------------------------------------------------------------


def _read(name: str, camp: Campaign, opaque: str, session: str) -> str:
    text = (FIXTURES / name).read_text("utf-8")
    return (text.replace("/campaign/exchange", str(camp.exchange_dir))
            .replace(F_OPAQUE, opaque).replace(F_SESSION, session))


def _lines(text: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in text.splitlines() if line]


def _dump(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(e) + "\n" for e in entries), "utf-8")


def synthetic(error: str, session: str) -> dict[str, Any]:
    return {"type": "assistant", "isSidechain": False, "sessionId": session, "error": error,
            "isApiErrorMessage": True,
            "message": {"id": "syn0001", "model": "<synthetic>", "role": "assistant",
                        "content": [{"type": "text", "text": "…"}]}}


def stage(camp: Campaign, session: str, opaque: str, shape: str = "clean", *,
          exit_code: int | None = 0) -> None:
    """Stage the driver transcript for `session` (plan6 §3.4's shapes) and the exit code."""
    driver = _lines(_read("driver.jsonl", camp, opaque, session))
    agent: list[dict[str, Any]] | None = _lines(_read(f"agent-{F_AGENT}.jsonl", camp, opaque,
                                                      session))
    meta: dict[str, Any] | None = json.loads(_read(f"agent-{F_AGENT}.meta.json", camp, opaque,
                                                   session))
    if shape == "no-tool-use":
        driver, agent, meta = driver[:1], None, None
    elif shape == "incomplete":
        driver = driver[:2]
        meta.pop("finished")
    elif shape == "rate-limit":
        driver, agent, meta = driver[:1] + [synthetic("rate_limit", session)], None, None
    elif shape == "opus-driver":
        for entry in driver:
            if entry.get("type") == "assistant":
                entry["message"]["model"] = "claude-opus-5-5"
    elif shape == "refusal":
        driver.append({"type": "user", "isSidechain": False, "sessionId": session,
                       "isMeta": True, "refusedMessageId": "msg_ref0001",
                       "message": {"role": "user", "content": "…"}})
    else:
        assert shape in ("clean", "missing"), shape
    if shape != "missing":
        path = camp.transcript_path(session)
        _dump(path, driver)
        subagents = path.with_suffix("") / "subagents"
        if agent is not None:
            _dump(subagents / f"agent-{F_AGENT}.jsonl", agent)
        if meta is not None:
            _dump(subagents / f"agent-{F_AGENT}.meta.json", [meta])
    if exit_code is not None:
        runs = camp.store / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        (runs / f"{session}.exit").write_text(f"{exit_code}\n", "utf-8")


def answer_bytes(camp: Campaign, opaque: str, *, answer: Any = True,
                 body: str | None = None) -> bytes:
    text = (camp.store / "req" / f"{opaque}.jsonl").read_text("utf-8")
    lines = text.splitlines()
    head = [{"bodySha256": body or json.loads(lines[0])["bodySha256"]}]
    entries = [{"key": json.loads(line)["key"], "answer": answer} for line in lines[2:]]
    return "".join(json.dumps(line) + "\n" for line in head + entries).encode("utf-8")


def write_answer(camp: Campaign, opaque: str, data: bytes, *, parts: int = 1) -> None:
    ans = camp.exchange_dir / "ans"
    if parts == 1:
        (ans / f"{opaque}.jsonl").write_bytes(data)
        return
    lines = data.splitlines(keepends=True)
    cut = len(lines) // 2
    (ans / f"{opaque}.1.jsonl").write_bytes(b"".join(lines[:cut]))
    (ans / f"{opaque}.2.jsonl").write_bytes(b"".join(lines[cut:]))


def serve(camp: Campaign) -> Serve:
    action = camp.next()
    assert isinstance(action, Serve), action
    return action


def run_batch(camp: Campaign, shape: str = "clean", *, answer: bytes | None = None,
              parts: int = 1, exit_code: int | None = 0, ingest: bool = True
              ) -> tuple[Serve, str]:
    """One batch through next → issue → (staged run) → ingest."""
    action = serve(camp)
    session = camp.issue(action)
    stage(camp, session, action.opaque, shape, exit_code=exit_code)
    if shape == "clean":
        write_answer(camp, action.opaque, answer or answer_bytes(camp, action.opaque),
                     parts=parts)
    if ingest:
        camp.ingest(action.opaque)
    return action, session


def exchange_files(camp: Campaign) -> list[str]:
    return sorted(p.name for folder in ("req", "ans")
                  for p in (camp.exchange_dir / folder).iterdir())


# --- serve and issue -----------------------------------------------------------------------------


def test_next_serves_the_pending_batch_with_the_spawn_prompt_on_an_absolute_path(
        tmp_path: Path) -> None:
    camp = smoke(tmp_path)
    action = serve(camp)
    request = camp.exchange_dir / "req" / f"{action.opaque}.jsonl"
    assert action.batch == "bat_0001" and action.agent_type == "dm-labeler"
    assert action.spawn_prompt == SPAWN_TEMPLATE.replace("{request_path}", str(request))
    assert os.path.isabs(str(request))
    assert exchange_files(camp) == []


def test_issue_records_the_session_id_then_writes_the_request(tmp_path: Path) -> None:
    camp = smoke(tmp_path)
    action = serve(camp)
    session = camp.issue(action, session_id="0f0f0f0f-0000-4000-8000-000000000001")
    (issued,) = [e for e in camp.events() if e["event"] == "issued"]
    assert issued["sessionId"] == session == "0f0f0f0f-0000-4000-8000-000000000001"
    assert issued["opaque"] == action.opaque and issued["batch"] == "bat_0001"
    assert issued["attempt"] == 1 and set(issued["keys"]) == set(action.request.keys)
    assert (camp.exchange_dir / "req" / f"{action.opaque}.jsonl").read_text("utf-8") \
        == (camp.store / "req" / f"{action.opaque}.jsonl").read_text("utf-8") \
        == action.request.text


def test_a_fresh_session_id_is_a_uuid4(tmp_path: Path) -> None:
    camp = smoke(tmp_path)
    session = camp.issue(serve(camp))
    assert len(session) == 36 and session[14] == "4"


def test_a_clean_run_ingests_and_moves_its_files_to_the_store(tmp_path: Path) -> None:
    camp = smoke(tmp_path)
    action, _ = run_batch(camp)
    (event,) = outcomes(camp)
    assert event["outcome"] == "ingested" and "reason" not in event
    assert event["answerSha256"] and event["run"]["runId"] == "run_0001"
    assert event["run"]["batchId"] == "bat_0001" and event["run"]["role"] == "label"
    assert exchange_files(camp) == []
    assert (camp.store / "ans" / f"{action.opaque}.jsonl").is_file()
    assert camp.state().batches["bat_0001"].state == "ingested"
    assert camp.next() == Stop("done", "nothing is pending")


def test_a_batch_past_the_cap_is_cut_and_the_overflow_queued(tmp_path: Path) -> None:
    camp = smoke(tmp_path, items=campaign.DEFAULT_LABEL_BATCH_CAP + 3)
    action = serve(camp)
    state = camp.state()
    assert len(state.batches["bat_0001"].entries) == campaign.DEFAULT_LABEL_BATCH_CAP
    assert state.batches["bat_0002"].entries == ["fx_0051", "fx_0052", "fx_0053"]
    assert len(action.request.keys) == campaign.DEFAULT_LABEL_BATCH_CAP


def test_a_generation_batch_is_served_from_recorded_seeds(tmp_path: Path) -> None:
    camp = smoke(tmp_path)
    draw = seeds.draw_seeds("guard-judge", 3, camp.fresh_rng_seed(), design_stratum=None)
    camp.record_draw(draw)
    run_batch(camp)
    batch = camp.queue_batch("generate", [seed.seed_id for seed in draw.seeds])
    action = serve(camp)
    assert action.batch == batch and action.agent_type == "dm-generator"
    assert {s.seed_id for s in action.request.keys.values()} == {s.seed_id for s in draw.seeds}


# --- the values later units read, each computed here (impl review P10-M1) -----------------------


def issued_events(camp: Campaign) -> list[dict[str, Any]]:
    return [e for e in camp.events() if e["event"] == "issued"]


def test_attempt_run_id_and_request_sha256_reach_the_run_record(tmp_path: Path) -> None:
    """plan6 §3.7: `attempt` is the batch's issue ordinal from 1, no-attempt re-issues included,
    so `runs[]` skips 1 here; `runId` numbers located runs only; `requestSha256` is the stored
    request's bytes (plan4 §3.7)."""
    camp = smoke(tmp_path)
    run_batch(camp, "missing", exit_code=1)         # launchFailed: no located run, no runId
    run_batch(camp, answer=b'{"bodySha256": "0"}\n')  # counted void
    action, _ = run_batch(camp)                     # ingested
    runs = [e["run"] for e in outcomes(camp)]
    assert [e["outcome"] for e in outcomes(camp)] == ["reissued", "void", "ingested"]
    assert runs[0] is None
    assert [e["attempt"] for e in issued_events(camp)] == [1, 2, 3]
    assert [(r["runId"], r["attempt"]) for r in runs[1:]] == [("run_0001", 2), ("run_0002", 3)]
    for event, run in zip(issued_events(camp), runs):
        stored = (camp.store / "req" / f"{event['opaque']}.jsonl").read_bytes()
        assert event["requestSha256"] == hashlib.sha256(stored).hexdigest()
        if run is not None:
            assert run["requestSha256"] == event["requestSha256"]
            assert run["batchId"] == "bat_0001"
    answer = (camp.store / "ans" / f"{action.opaque}.jsonl").read_bytes()
    assert outcomes(camp)[-1]["answerSha256"] == hashlib.sha256(answer).hexdigest()


def test_issued_keys_bind_each_key_to_the_item_its_request_line_renders(tmp_path: Path) -> None:
    """plan4 §3.2: the binding is the only record of which item an answer key labels."""
    camp = smoke(tmp_path, items=3)
    action = serve(camp)
    camp.issue(action)
    (event,) = issued_events(camp)
    lines = [json.loads(line) for line in action.request.text.splitlines()[2:]]
    assert sorted(event["keys"].values()) == ["fx_0001", "fx_0002", "fx_0003"]
    render = exchange.RENDERERS["guard-judge"]
    for line in lines:
        ref = event["keys"][line["key"]]
        assert line["text"] == render(guard_item(int(ref.removeprefix("fx_")) - 1))


def test_issued_keys_bind_each_key_to_the_seed_its_request_line_carries(tmp_path: Path) -> None:
    camp = smoke(tmp_path)
    run_batch(camp)
    draw = seeds.draw_seeds("guard-judge", 4, 9, design_stratum=None)
    camp.record_draw(draw)
    camp.queue_batch("generate", [seed.seed_id for seed in draw.seeds])
    action = serve(camp)
    camp.issue(action)
    event = issued_events(camp)[-1]
    lines = [json.loads(line) for line in action.request.text.splitlines()[2:]]
    bodies = [{k: v for k, v in line.items() if k != "key"} for line in lines]
    assert len({json.dumps(b, sort_keys=True) for b in bodies}) == len(bodies) > 1
    by_id = {seed.seed_id: seed for seed in draw.seeds}
    for line, body in zip(lines, bodies):
        axes = by_id[event["keys"][line["key"]]].axes
        assert body and all(axes[name] == value for name, value in body.items())


def test_a_generation_run_record_is_role_generate(tmp_path: Path) -> None:
    camp = smoke(tmp_path)
    run_batch(camp)
    draw = seeds.draw_seeds("guard-judge", 2, 5, design_stratum=None)
    camp.record_draw(draw)
    camp.queue_batch("generate", [s.seed_id for s in draw.seeds])
    action = serve(camp)
    session = camp.issue(action)
    stage(camp, session, action.opaque)             # located (the label fixture: it halts)
    write_answer(camp, action.opaque, b"")
    camp.ingest(action.opaque)
    run = outcomes(camp)[-1]["run"]
    assert (run["role"], run["batchId"], run["attempt"]) == ("generate", "bat_0002", 1)


def test_an_outcome_records_its_details_and_counts_through_void_alone(tmp_path: Path) -> None:
    """`details` is the check's own text; a counted attempt is `outcome: "void"` and nothing else
    (P10-n1: the event carries no second copy, `countsAttempt`, for `fold` to disagree with)."""
    camp = smoke(tmp_path)
    bad = b'{"bodySha256": "0"}\n'
    action, _ = run_batch(camp, answer=bad)
    void = outcomes(camp)[-1]
    expected = ingest.ingest_answer(action.request.text, [bad], ingested_answer_sha256=None)
    assert void["details"] == list(expected.details) and void["details"]
    assert void["answerSha256"] == hashlib.sha256(bad).hexdigest()
    assert camp.state().batches["bat_0001"].counted == 1
    run_batch(camp, "opus-driver")
    halted = outcomes(camp)[-1]
    assert halted["details"] == ["driver message.model ['claude-opus-5-5'] is not "
                                 "claude-sonnet-5-5"]
    assert all("countsAttempt" not in event for event in outcomes(camp))
    assert camp.state().batches["bat_0001"].counted == 1


def test_issue_refuses_a_stale_request_and_writes_nothing(tmp_path: Path) -> None:
    camp = smoke(tmp_path)
    camp.queue_batch("label", ["fx_0001"])
    camp.queue_batch("label", ["fx_0002"])
    built = camp.prepare("bat_0002")
    before = ledger(camp)
    with pytest.raises(CampaignRefused, match="stale for bat_0003"):
        camp.issue(dataclasses.replace(built, batch="bat_0003"))
    assert ledger(camp) == before and not (camp.store / "req").exists()
    assert exchange_files(camp) == []


def test_issue_refuses_an_opaque_name_already_issued(tmp_path: Path) -> None:
    """A re-issued batch is pending again, but its retired name is never issued twice."""
    camp = smoke(tmp_path)
    action, _ = run_batch(camp, "no-tool-use")
    assert camp.state().batches["bat_0001"].state == "pending"
    before = ledger(camp)
    with pytest.raises(CampaignRefused, match=f"{action.opaque} is already issued"):
        camp.issue(action)
    assert ledger(camp) == before and exchange_files(camp) == []


def test_next_needs_declared_expectations(tmp_path: Path) -> None:
    fixture = write_fixture(tmp_path / "fixture.jsonl", 1)
    camp = Campaign(tmp_path / "camp", claude_dir=tmp_path / "claude", clock=Clock())
    camp.init(point="guard-judge", phase="smoke", fixture=fixture)
    with pytest.raises(CampaignRefused, match="expectations"):
        camp.next()


# --- plan4 §5 U5.6: resume -----------------------------------------------------------------------


def test_resume_ingests_reissues_serves_and_never_returns_an_ingested_batch(
        tmp_path: Path) -> None:
    """Batches ingested, issued (complete transcript), issued (no `tool_use`), pending."""
    camp = smoke(tmp_path)
    for _ in range(3):
        camp.queue_batch("label", ["fx_0001", "fx_0002"])
    run_batch(camp)                                     # bat_0001 ingested
    second = camp.prepare("bat_0002")
    s2 = camp.issue(second)
    stage(camp, s2, second.opaque)
    write_answer(camp, second.opaque, answer_bytes(camp, second.opaque))
    third = camp.prepare("bat_0003")
    s3 = camp.issue(third)
    stage(camp, s3, third.opaque, "no-tool-use")

    action = serve(camp)
    state = camp.state()
    assert state.batches["bat_0002"].state == "ingested"
    assert action.batch == "bat_0003" and action.opaque != third.opaque
    assert state.batches["bat_0003"].counted == 0
    assert exchange_files(camp) == []                   # before the re-issue
    reissue = state.opaques[third.opaque].outcome
    assert reissue["outcome"] == "reissued" and reissue["reason"] == "neverSpawned"

    session = camp.issue(action)
    stage(camp, session, action.opaque)
    write_answer(camp, action.opaque, answer_bytes(camp, action.opaque))
    assert serve(camp).batch == "bat_0004"
    served = [e["batch"] for e in camp.events() if e["event"] == "issued"]
    assert served.count("bat_0001") == 1


# --- plan4 §5 U5.6: caps -------------------------------------------------------------------------


def test_two_counted_voids_reissue_and_the_third_drops_as_retry_cap_exceeded(
        tmp_path: Path) -> None:
    camp = smoke(tmp_path)
    for attempt in range(1, 4):
        action, _ = run_batch(camp, answer=b'{"bodySha256": "0"}\n')
        batch = camp.state().batches["bat_0001"]
        assert outcomes(camp)[-1]["outcome"] == "void"
        assert outcomes(camp)[-1]["reason"] == "bodyHashMismatch"
        assert batch.counted == attempt
        assert batch.state == ("pending" if attempt < 3 else "dropped")
    state = camp.state()
    assert state.batches["bat_0001"].dropped == "retryCapExceeded"
    assert state.dropped()["retryCapExceeded"] == {"seeds": 0, "items": 2}
    assert camp.next() == Stop("done", "nothing is pending")


def test_an_environmental_void_halts_and_counts_nothing(tmp_path: Path) -> None:
    camp = smoke(tmp_path)
    run_batch(camp, "opus-driver")
    batch = camp.state().batches["bat_0001"]
    assert outcomes(camp)[-1]["outcome"] == "halted" and batch.counted == 0
    assert outcomes(camp)[-1]["reason"] == "driverModelMismatch"
    assert "answerSha256" not in outcomes(camp)[-1]


def test_ten_usage_stops_count_nothing_and_never_halt(tmp_path: Path) -> None:
    camp = smoke(tmp_path)
    for _ in range(10):
        camp = fresh(camp)                              # one run per spent window
        action = serve(camp)
        session = camp.issue(action)
        stage(camp, session, action.opaque, "rate-limit")
        result = camp.ingest(action.opaque)
        assert result.stop == "usage-stop"
        assert result.outcome["outcome"] == "usage-stop"
    state = camp.state()
    assert state.halt is None and state.batches["bat_0001"].counted == 0
    assert state.batches["bat_0001"].state == "pending"


def test_next_stops_on_a_usage_stop_its_reconciliation_meets(tmp_path: Path) -> None:
    camp = smoke(tmp_path)
    action = serve(camp)
    session = camp.issue(action)
    stage(camp, session, action.opaque, "rate-limit")
    before = ledger(camp)
    assert camp.next() == Stop("usage-stop", "rateLimit")
    assert camp.next() == Stop("usage-stop", "rateLimit")   # this run launches nothing more
    assert len(ledger(camp)) > len(before) and outcomes(camp)[-1]["outcome"] == "usage-stop"
    assert isinstance(fresh(camp).next(), Serve)        # the next run serves again


def test_a_usage_stop_run_ingests_ends_the_runner_loop_after_one_launch(tmp_path: Path) -> None:
    """P10-m1 (b): plan6 §3.2's loop, as written — `next`, `issue`, launch, `ingest`, loop until
    `next` returns stop — launches exactly once when the window is spent."""
    camp = smoke(tmp_path, items=3)
    camp.queue_batch("label", ["fx_0003"])
    launches = []
    while isinstance(action := camp.next(), Serve):
        session = camp.issue(action)
        launches.append(action.opaque)                  # the child: it meets the spent window
        stage(camp, session, action.opaque, "rate-limit")
        assert camp.ingest(action.opaque).stop == "usage-stop"
        assert len(launches) < 3, "the loop re-launched inside a spent window"
    assert action == Stop("usage-stop", "rateLimit") and len(launches) == 1
    assert [e["outcome"] for e in outcomes(camp)] == ["usage-stop"]
    assert camp.state().batches["bat_0001"].state == "pending"
    again = fresh(camp)
    assert again.usage_stop is None and serve(again).batch == "bat_0001"


def test_a_re_ingest_of_a_usage_stop_is_not_a_new_usage_stop(tmp_path: Path) -> None:
    """Only a usage stop this instance recorded stops it; re-reading the ledger's does not."""
    camp = smoke(tmp_path)
    action = serve(camp)
    session = camp.issue(action)
    stage(camp, session, action.opaque, "rate-limit")
    camp.ingest(action.opaque)
    again = fresh(camp)
    assert again.ingest(action.opaque).stop is None
    assert isinstance(again.next(), Serve)


@pytest.mark.parametrize("shape, reason", [("no-tool-use", "neverSpawned"),
                                           ("incomplete", "incomplete")])
def test_the_second_never_spawned_or_incomplete_outcome_for_one_batch_halts(
        tmp_path: Path, shape: str, reason: str) -> None:
    camp = smoke(tmp_path)
    run_batch(camp, shape)
    assert outcomes(camp)[-1]["outcome"] == "reissued"
    run_batch(camp, shape)
    last = outcomes(camp)[-1]
    assert (last["outcome"], last["reason"], last["voidClass"]) == ("halted", reason, "driver")
    assert camp.state().batches["bat_0001"].counted == 0


def test_never_spawned_and_incomplete_share_one_cap(tmp_path: Path) -> None:
    camp = smoke(tmp_path)
    run_batch(camp, "no-tool-use")
    run_batch(camp, "incomplete")
    assert outcomes(camp)[-1]["outcome"] == "halted"


def test_after_a_halt_next_refuses_until_resolve(tmp_path: Path) -> None:
    camp = smoke(tmp_path)
    run_batch(camp, "opus-driver")
    before = ledger(camp)
    for _ in range(2):
        stop = camp.next()
        assert isinstance(stop, Stop) and stop.reason == "halted"
        assert "driverModelMismatch" in stop.detail
    assert ledger(camp) == before
    camp.resolve(decision="the driver pin was fixed")
    action = serve(camp)
    assert action.batch == "bat_0001"


def test_resolve_restarts_the_no_attempt_cap(tmp_path: Path) -> None:
    camp = smoke(tmp_path)
    run_batch(camp, "incomplete")
    run_batch(camp, "incomplete")
    camp.resolve(decision="killed by hand twice; retry")
    run_batch(camp, "incomplete")
    assert outcomes(camp)[-1]["outcome"] == "reissued"


# --- plan6 §3.2: interrupted, launches, the session id -------------------------------------------


def test_an_interrupted_incomplete_does_not_count_toward_the_cap(tmp_path: Path) -> None:
    """plan6 §5 (Interrupted) as the campaign applies it, and Pass 9's carry: an interrupted
    outcome stays out of `earlier_no_attempt`, so a later plain incomplete re-issues."""
    camp = smoke(tmp_path)
    action = serve(camp)
    session = camp.issue(action)
    stage(camp, session, action.opaque, "incomplete")
    camp.record_interrupted("bat_0001")
    camp.ingest(action.opaque)
    assert outcomes(camp)[-1]["outcome"] == "reissued"
    run_batch(camp, "incomplete")
    assert outcomes(camp)[-1]["outcome"] == "reissued"
    run_batch(camp, "incomplete")
    assert outcomes(camp)[-1]["outcome"] == "halted"


def test_a_second_incomplete_named_by_interrupted_does_not_halt(tmp_path: Path) -> None:
    camp = smoke(tmp_path)
    run_batch(camp, "incomplete")
    action = serve(camp)
    session = camp.issue(action)
    stage(camp, session, action.opaque, "incomplete")
    camp.record_interrupted("bat_0001")
    camp.ingest(action.opaque)
    assert outcomes(camp)[-1]["outcome"] == "reissued"


def test_interrupted_needs_an_issued_attempt(tmp_path: Path) -> None:
    camp = smoke(tmp_path)
    with pytest.raises(CampaignRefused, match="no issued attempt"):
        camp.record_interrupted("bat_0001")


def test_reconciliation_reads_only_the_recorded_session(tmp_path: Path) -> None:
    """plan6 §5 U5.6: a second driver session in the same project folder is ignored."""
    camp = smoke(tmp_path)
    action = serve(camp)
    session = camp.issue(action)
    stage(camp, "0f0f0f0f-0000-4000-8000-0000000000ff", action.opaque)   # planted, clean
    write_answer(camp, action.opaque, answer_bytes(camp, action.opaque))
    stage(camp, session, action.opaque, "missing", exit_code=1)
    camp.ingest(action.opaque)
    last = outcomes(camp)[-1]
    assert (last["outcome"], last["reason"]) == ("reissued", "launchFailed")
    assert "answerSha256" not in last


def test_a_launch_failure_records_stdout_stderr_and_the_exit_code(tmp_path: Path) -> None:
    camp = smoke(tmp_path)
    action = serve(camp)
    session = camp.issue(action)
    stage(camp, session, action.opaque, "missing", exit_code=127)
    (camp.store / "runs" / f"{session}.err").write_text("claude: not found\n", "utf-8")
    camp.ingest(action.opaque)
    last = outcomes(camp)[-1]
    assert last["launch"] == {"exitCode": 127, "stdout": None, "stderr": "claude: not found\n"}
    assert last["produced"] is False and last["run"] is None


def test_a_prompt_only_transcript_with_an_unobserved_exit_is_a_launch_failure(
        tmp_path: Path) -> None:
    camp = smoke(tmp_path)
    run_batch(camp, "no-tool-use", exit_code=None)
    last = outcomes(camp)[-1]
    assert last["reason"] == "launchFailed" and last["launch"]["exitCode"] is None


@pytest.mark.parametrize("content", [b"", b"\n", b"-", b"abc\n", b"1.5\n", b"+1\n", b"1_0\n",
                                     b"\xff\xfe", "\u0661\n".encode("utf-8")])
def test_an_exit_file_not_in_the_format_is_an_exit_never_observed(
        tmp_path: Path, content: bytes) -> None:
    """P10-m2: an empty file (a runner killed mid-write) or any non-decimal text reads as a
    missing one — `launchFailed` with `exitCode: null` — and never raises from `next`."""
    camp = smoke(tmp_path)
    action = serve(camp)
    session = camp.issue(action)
    stage(camp, session, action.opaque, "no-tool-use", exit_code=None)
    (camp.store / "runs").mkdir(parents=True, exist_ok=True)
    (camp.store / "runs" / f"{session}.exit").write_bytes(content)
    assert isinstance(camp.next(), Serve)
    last = outcomes(camp)[-1]
    assert (last["outcome"], last["reason"]) == ("reissued", "launchFailed")
    assert last["launch"]["exitCode"] is None


@pytest.mark.parametrize("content, code", [(b"-15\n", -15), (b" 3 \n", 3), (b"0", 0)])
def test_an_exit_file_in_the_format_is_read(tmp_path: Path, content: bytes, code: int) -> None:
    camp = smoke(tmp_path)
    action = serve(camp)
    session = camp.issue(action)
    stage(camp, session, action.opaque, "no-tool-use", exit_code=None)
    (camp.store / "runs").mkdir(parents=True, exist_ok=True)
    (camp.store / "runs" / f"{session}.exit").write_bytes(content)
    camp.ingest(action.opaque)
    last = outcomes(camp)[-1]
    if code == 0:                                       # plan6 §3.4: exit 0 → never spawned
        assert (last["reason"], "launch" in last) == ("neverSpawned", False)
    else:
        assert (last["reason"], last["launch"]["exitCode"]) == ("launchFailed", code)


def launch_failure_at(camp: Campaign, clock: Clock, at: datetime) -> dict[str, Any]:
    clock.now = at - timedelta(seconds=1)
    run_batch(camp, "missing", exit_code=1)
    return outcomes(camp)[-1]


@pytest.mark.parametrize("gap, reason", [(timedelta(minutes=59), "launchFailed"),
                                         (timedelta(minutes=61), "launchFailing")])
def test_launch_failing_follows_the_bound_recorded_at_init(
        tmp_path: Path, gap: timedelta, reason: str) -> None:
    """`Launch.fail_bound` is `init`'s `launchFailBound` (1 h here), and the clock reads the
    ledger's `issued` times."""
    clock = Clock()
    camp = smoke(tmp_path, clock=clock, bound=timedelta(hours=1))
    first = launch_failure_at(camp, clock, T0 + timedelta(hours=1))
    assert first["reason"] == "launchFailed"
    launch_failure_at(camp, clock, T0 + timedelta(hours=1, minutes=30))
    last = launch_failure_at(camp, clock, T0 + timedelta(hours=1) + gap)
    assert last["reason"] == reason
    assert last["outcome"] == ("reissued" if reason == "launchFailed" else "halted")


def test_a_launch_failing_halt_keeps_its_launch_evidence(tmp_path: Path) -> None:
    """plan6 §3.4: at a `launchFailing` halt the evidence is what the human reads (P10-n1)."""
    clock = Clock()
    camp = smoke(tmp_path, clock=clock, bound=timedelta(hours=1))
    launch_failure_at(camp, clock, T0 + timedelta(hours=1))
    clock.now = T0 + timedelta(hours=2, minutes=1)
    action = serve(camp)
    session = camp.issue(action)
    stage(camp, session, action.opaque, "missing", exit_code=2)
    (camp.store / "runs" / f"{session}.json").write_text('{"is_error": true}', "utf-8")
    (camp.store / "runs" / f"{session}.err").write_text("auth expired\n", "utf-8")
    camp.ingest(action.opaque)
    last = outcomes(camp)[-1]
    assert (last["outcome"], last["reason"]) == ("halted", "launchFailing")
    assert last["launch"] == {"exitCode": 2, "stdout": '{"is_error": true}',
                              "stderr": "auth expired\n"}


def test_resolve_restarts_the_launch_fail_clock(tmp_path: Path) -> None:
    """P10-m4 (the implementer's call 2): a `launchFailing` halt, then `resolve`, then one more
    launch failure past the bound from the first → re-issued, not halted again."""
    clock = Clock()
    camp = smoke(tmp_path, clock=clock, bound=timedelta(hours=1))
    first = T0 + timedelta(hours=1)
    launch_failure_at(camp, clock, first)
    halted = launch_failure_at(camp, clock, first + timedelta(minutes=61))
    assert (halted["outcome"], halted["reason"]) == ("halted", "launchFailing")
    assert halted["launch"]["exitCode"] == 1
    camp.resolve(decision="the CLI was reinstalled; retry")
    last = launch_failure_at(camp, clock, first + timedelta(minutes=62))
    assert (last["outcome"], last["reason"]) == ("reissued", "launchFailed")


def test_a_launch_that_produced_something_restarts_the_clock(tmp_path: Path) -> None:
    clock = Clock()
    camp = smoke(tmp_path, clock=clock, bound=timedelta(hours=1))
    launch_failure_at(camp, clock, T0 + timedelta(hours=1))
    clock.now = T0 + timedelta(hours=1, minutes=30)
    action = serve(camp)
    session = camp.issue(action)
    stage(camp, session, action.opaque, "rate-limit")
    camp.ingest(action.opaque)
    last = launch_failure_at(fresh(camp), clock, T0 + timedelta(hours=2, minutes=1))
    assert last["reason"] == "launchFailed"


def test_expected_paths_are_absolute_under_a_relative_root(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pass 9: relative paths would make every run `guardBreach`."""
    monkeypatch.chdir(tmp_path)
    camp = smoke(tmp_path, root=Path("camp"))
    expected = camp._expected(camp.state(), camp.state().batches["bat_0001"], "x" * 16)
    assert os.path.isabs(expected.request_path) and os.path.isabs(expected.answer_path)
    run_batch(camp)
    assert outcomes(camp)[-1]["outcome"] == "ingested"


# --- the ingest commit point and re-ingest (coordination ledger, P5-F4, P6-F1, Pass 8 q3) ---------


def crash_on_move(monkeypatch: pytest.MonkeyPatch, *, after_copies: int | None = None,
                  during_unlink: bool = False) -> None:
    """Make the next `_move_to_store` die: before anything moves, after `after_copies` copies,
    or after every copy and one unlink."""
    real_write = campaign._write_atomic
    real_unlink = Path.unlink
    copies = {"n": 0}

    def write(path: Path, data: bytes) -> None:
        if path.parent.name == "ans":
            if after_copies is not None and copies["n"] >= after_copies:
                raise OSError("killed mid-move")
            copies["n"] += 1
        real_write(path, data)

    def unlink(self: Path, missing_ok: bool = False) -> None:
        real_unlink(self, missing_ok=missing_ok)
        if during_unlink and self.parent.name == "ans":
            raise OSError("killed mid-unlink")

    if after_copies is None and not during_unlink:
        def die(*args: Any, **kwargs: Any) -> None:
            raise OSError("killed before the move")
        monkeypatch.setattr(Campaign, "_move_to_store", die)
        return
    monkeypatch.setattr(campaign, "_write_atomic", write)
    monkeypatch.setattr(Path, "unlink", unlink)


def test_case_a_a_repeat_ingest_with_parts_already_in_the_store_is_a_no_op(
        tmp_path: Path) -> None:
    camp = smoke(tmp_path)
    action, _ = run_batch(camp)
    before = ledger(camp)
    result = camp.ingest(action.opaque)
    assert result.outcome is None and result.stop is None
    assert ledger(camp) == before
    assert (camp.store / "ans" / f"{action.opaque}.jsonl").is_file()


def test_case_b_a_crash_after_the_event_before_the_move_re_ingests_as_a_no_op(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = smoke(tmp_path)
    with monkeypatch.context() as patch:
        crash_on_move(patch)
        with pytest.raises(OSError, match="before the move"):
            run_batch(camp)
    (event,) = outcomes(camp)                           # the event is the commit point
    assert event["outcome"] == "ingested"
    opaque = event["opaque"]
    assert (camp.exchange_dir / "ans" / f"{opaque}.jsonl").is_file()
    before = ledger(camp)
    assert isinstance(camp.next(), Stop)                # reconciles: finishes the move
    assert ledger(camp) == before
    assert exchange_files(camp) == []
    assert (camp.store / "ans" / f"{opaque}.jsonl").is_file()


def test_case_c_a_crashed_void_re_ingests_as_one_counted_attempt(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = smoke(tmp_path)
    with monkeypatch.context() as patch:
        crash_on_move(patch)
        with pytest.raises(OSError):
            run_batch(camp, answer=b'{"bodySha256": "0"}\n')
    (event,) = outcomes(camp)
    assert event["outcome"] == "void" and event["answerSha256"]
    before = ledger(camp)
    camp.ingest(event["opaque"])
    assert ledger(camp) == before
    assert camp.state().batches["bat_0001"].counted == 1


def test_case_d_a_different_answer_for_an_ingested_name_is_refused(tmp_path: Path) -> None:
    camp = smoke(tmp_path)
    action, _ = run_batch(camp)
    write_answer(camp, action.opaque, answer_bytes(camp, action.opaque, answer=False))
    before = ledger(camp)
    with pytest.raises(AnswerRefused):
        camp.ingest(action.opaque)
    with pytest.raises(AnswerRefused):
        camp.next()
    assert ledger(camp) == before


def test_case_d_a_different_answer_before_the_move_is_refused(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = smoke(tmp_path)
    with monkeypatch.context() as patch:
        crash_on_move(patch)
        with pytest.raises(OSError):
            run_batch(camp)
    opaque = outcomes(camp)[0]["opaque"]
    write_answer(camp, opaque, answer_bytes(camp, opaque, answer=False))
    before = ledger(camp)
    with pytest.raises(AnswerRefused):
        camp.ingest(opaque)
    assert ledger(camp) == before


@pytest.mark.parametrize("crash", [{"after_copies": 1}, {"during_unlink": True}])
def test_case_e_a_crash_mid_move_re_ingests_as_a_no_op(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, crash: dict[str, Any]) -> None:
    """A two-part answer; the set move dies with part 1 copied, or after one unlink (part 1 gone
    from `exchange/`, both in `store/`)."""
    camp = smoke(tmp_path, items=4)
    with monkeypatch.context() as patch:
        crash_on_move(patch, **crash)
        with pytest.raises(OSError):
            run_batch(camp, parts=2)
    (event,) = outcomes(camp)
    assert event["outcome"] == "ingested" and len(event["parts"]) == 2
    before = ledger(camp)
    camp.ingest(event["opaque"])
    assert ledger(camp) == before
    assert exchange_files(camp) == []
    assert sorted(p.name for p in (camp.store / "ans").iterdir()) == sorted(event["parts"])


def test_the_re_ingest_path_never_calls_the_audit_or_combine(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = smoke(tmp_path)
    with monkeypatch.context() as patch:
        crash_on_move(patch)
        with pytest.raises(OSError):
            run_batch(camp)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the re-ingest path called the audit")

    monkeypatch.setattr(audit, "audit_run", forbidden)
    monkeypatch.setattr(audit, "combine", forbidden)
    before = ledger(camp)
    camp.ingest(outcomes(camp)[0]["opaque"])
    assert ledger(camp) == before


@pytest.mark.parametrize("shape", ["no-tool-use", "rate-limit", "incomplete"])
def test_a_late_answer_to_a_retired_name_is_refused(tmp_path: Path, shape: str) -> None:
    """Architect reconciliation of Pass 8: a re-issued name is retired."""
    camp = smoke(tmp_path)
    action, _ = run_batch(camp, shape)
    assert "answerSha256" not in outcomes(camp)[-1]
    write_answer(camp, action.opaque, answer_bytes(camp, action.opaque))
    before = ledger(camp)
    with pytest.raises(AnswerRefused, match="did not record"):
        camp.ingest(action.opaque)
    with pytest.raises(AnswerRefused):
        camp.next()
    assert ledger(camp) == before
    assert camp.state().batches["bat_0001"].state == "pending"


def test_an_answer_part_written_before_a_usage_stop_moves_with_the_attempt(
        tmp_path: Path) -> None:
    camp = smoke(tmp_path)
    action = serve(camp)
    session = camp.issue(action)
    stage(camp, session, action.opaque, "rate-limit")
    write_answer(camp, action.opaque, answer_bytes(camp, action.opaque))
    camp.ingest(action.opaque)
    assert outcomes(camp)[-1]["parts"] == [f"{action.opaque}.jsonl"]
    assert exchange_files(camp) == []
    before = ledger(camp)
    camp.ingest(action.opaque)
    assert ledger(camp) == before


def test_a_file_no_attempt_accounts_for_is_refused(tmp_path: Path) -> None:
    camp = smoke(tmp_path)
    (camp.exchange_dir / "ans" / "stranger.jsonl").write_text("{}\n", "utf-8")
    with pytest.raises(CampaignRefused, match="no attempt accounts for"):
        camp.next()


def test_ingest_of_an_unknown_name_is_refused(tmp_path: Path) -> None:
    camp = smoke(tmp_path)
    with pytest.raises(CampaignRefused, match="never issued"):
        camp.ingest("abcdefghijklmnop")


# --- plan6 §5 U5.6: the stop switch --------------------------------------------------------------


def test_next_stops_while_paused_and_serves_again_once_resumed(tmp_path: Path) -> None:
    camp = smoke(tmp_path)
    action = serve(camp)
    session = camp.issue(action)
    stage(camp, session, action.opaque)                 # an issued batch awaiting reconcile
    write_answer(camp, action.opaque, answer_bytes(camp, action.opaque))
    camp.queue_batch("label", ["fx_0001"])
    camp.pause_path.touch()
    before = ledger(camp)
    for _ in range(3):
        assert camp.next() == Stop("paused", "store/pause exists")
    assert ledger(camp) == before
    camp.pause_path.unlink()
    assert serve(camp).batch == "bat_0002"


# --- init ----------------------------------------------------------------------------------------


def test_init_records_batch_timeout_and_launch_fail_bound(tmp_path: Path) -> None:
    camp = Campaign(tmp_path / "camp", clock=Clock())
    camp.init(point="guard-judge", phase="smoke", fixture=write_fixture(tmp_path / "f.jsonl", 1),
              batch_timeout=timedelta(minutes=30), launch_fail_bound=timedelta(hours=8))
    init = camp.events()[0]
    assert (init["event"], init["batchTimeout"], init["launchFailBound"]) == ("init", 1800, 28800)
    assert (camp.exchange_dir / "empty-mcp.json").is_file()
    assert (camp.exchange_dir / "req").is_dir() and (camp.exchange_dir / "ans").is_dir()


def test_init_defaults_are_45_minutes_and_6_hours(tmp_path: Path) -> None:
    camp = Campaign(tmp_path / "camp", clock=Clock())
    camp.init(point="guard-judge", phase="pilot")
    init = camp.events()[0]
    assert (init["batchTimeout"], init["launchFailBound"]) == (2700, 21600)
    assert init["labelBatchCap"] == 50
    assert camp.state().batches == {}


def refused_init(root: Path, match: str, **kwargs: Any) -> None:
    camp = Campaign(root, clock=Clock())
    with pytest.raises(CampaignRefused, match=match):
        camp.init(**{"point": "guard-judge", "phase": "pilot", **kwargs})
    assert not (root / "store" / "ledger.jsonl").exists()


@pytest.mark.parametrize("marker", [".git", "CLAUDE.md", "AGENTS.md"])
def test_init_refuses_a_root_under_a_work_tree_or_a_context_file(
        tmp_path: Path, marker: str) -> None:
    (tmp_path / "outer").mkdir()
    (tmp_path / "outer" / marker).mkdir() if marker == ".git" else \
        (tmp_path / "outer" / marker).write_text("x", "utf-8")
    refused_init(tmp_path / "outer" / "deep" / "camp",
                 "git work tree" if marker == ".git" else marker)


def test_a_symlinked_root_into_a_work_tree_is_refused(tmp_path: Path) -> None:
    """P10-m3: the root is the physical path, so a symlink cannot hide the work tree."""
    (tmp_path / "tree" / ".git").mkdir(parents=True)
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "link").symlink_to(tmp_path / "tree")
    refused_init(tmp_path / "outside" / "link" / "camp", "git work tree")
    assert not (tmp_path / "tree" / "camp").exists()


def test_a_symlinked_root_names_the_physical_transcript_folder(tmp_path: Path) -> None:
    """The child started with `cwd=` the exchange folder sees its physical path (`getcwd`)."""
    (tmp_path / "real").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "real")
    camp = smoke(tmp_path, root=tmp_path / "link" / "camp")
    physical = tmp_path.resolve() / "real" / "camp" / "exchange"
    assert camp.root == tmp_path.resolve() / "real" / "camp"
    slug = "".join(c if c.isascii() and c.isalnum() else "-" for c in str(physical))
    assert camp.transcript_path("s") == tmp_path / "claude" / "projects" / slug / "s.jsonl"
    run_batch(camp)
    assert outcomes(camp)[-1]["outcome"] == "ingested"


def test_init_refuses_a_git_file_too(tmp_path: Path) -> None:
    """A worktree or submodule marks itself with a `.git` file."""
    (tmp_path / ".git").write_text("gitdir: elsewhere\n", "utf-8")
    refused_init(tmp_path / "camp", "git work tree")


def test_init_refuses_a_non_empty_root(tmp_path: Path) -> None:
    (tmp_path / "camp").mkdir()
    (tmp_path / "camp" / "x").write_text("x", "utf-8")
    refused_init(tmp_path / "camp", "not empty")


def test_smoke_needs_a_fixture_and_only_smoke_takes_one(tmp_path: Path) -> None:
    refused_init(tmp_path / "a", "--fixture", phase="smoke")
    refused_init(tmp_path / "b", "--fixture", phase="pilot",
                 fixture=write_fixture(tmp_path / "f.jsonl", 1))


def test_a_smoke_fixture_item_out_of_schema_is_refused(tmp_path: Path) -> None:
    fixture = tmp_path / "f.jsonl"
    fixture.write_text('{"condition": "x"}\n', "utf-8")
    refused_init(tmp_path / "camp", "line 1", phase="smoke", fixture=fixture)


def test_the_pilot_phase_is_guards(tmp_path: Path) -> None:
    refused_init(tmp_path / "camp", "pilot phase", point="join-in-judge")


def test_unknown_point_or_phase_is_refused(tmp_path: Path) -> None:
    refused_init(tmp_path / "a", "unknown point", point="nope")
    refused_init(tmp_path / "b", "unknown phase", phase="later")


def pilot_report(data_dir: Path, **report: Any) -> None:
    path = data_dir / "guard-judge" / "pilot" / "pilot-report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report), "utf-8")


PROCEED = {"outcome": "proceed", "cap": 25}


@pytest.mark.parametrize("decision, state", [
    (PROCEED, "passed"),
    (PROCEED, "recheck-passed"),
    ({**PROCEED, "stakeholderRef": "ledger line 12"}, "recheck-failed"),
])
def test_the_spend_gate_admits_each_passing_branch(
        tmp_path: Path, decision: dict[str, Any], state: str) -> None:
    data = tmp_path / "data"
    pilot_report(data, stakeholderDecision=decision, missingCueCheck={"state": state},
                 rngSeeds=[11, 12])
    camp = Campaign(tmp_path / "camp", clock=Clock())
    camp.init(point="guard-judge", phase="full", data_dir=data)
    assert camp.events()[0]["labelBatchCap"] == 25


@pytest.mark.parametrize("decision, state, match", [
    ({"outcome": "stop", "cap": 50}, "passed", "'stop'"),
    (PROCEED, "recheck-failed", "stakeholderRef"),
    (PROCEED, "failed", "'failed'"),
    (PROCEED, "recheck-pending", "'recheck-pending'"),
    (None, "passed", "no recorded stakeholderDecision"),
])
def test_the_spend_gate_refuses_each_failing_branch(
        tmp_path: Path, decision: Any, state: str, match: str) -> None:
    data = tmp_path / "data"
    pilot_report(data, stakeholderDecision=decision, missingCueCheck={"state": state})
    refused_init(tmp_path / "camp", match, phase="full", data_dir=data)


def refused_full(tmp_path: Path, match: str, report: Any) -> None:
    """`init --phase full` refuses and creates no campaign root (P10-M2)."""
    path = tmp_path / "data" / "guard-judge" / "pilot" / "pilot-report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report if isinstance(report, str) else json.dumps(report), "utf-8")
    refused_init(tmp_path / "camp", match, phase="full", data_dir=tmp_path / "data")
    assert not (tmp_path / "camp").exists()


GOOD_REPORT = {"stakeholderDecision": PROCEED, "missingCueCheck": {"state": "passed"},
               "rngSeeds": [11, 12]}


@pytest.mark.parametrize("cap", [None, "25", 25.0, True, 0, 40, 100])
def test_full_refuses_a_cap_that_is_absent_or_not_25_or_50(tmp_path: Path, cap: Any) -> None:
    decision = {"outcome": "proceed"} if cap is None else {"outcome": "proceed", "cap": cap}
    refused_full(tmp_path, "stakeholderDecision.cap", {**GOOD_REPORT,
                                                       "stakeholderDecision": decision})


def test_full_refuses_a_misnamed_cap_key(tmp_path: Path) -> None:
    refused_full(tmp_path, "stakeholderDecision.cap", {
        **GOOD_REPORT, "stakeholderDecision": {"outcome": "proceed", "capChosen": 25}})


@pytest.mark.parametrize("report", [
    {k: v for k, v in GOOD_REPORT.items() if k != "rngSeeds"},
    {**{k: v for k, v in GOOD_REPORT.items() if k != "rngSeeds"}, "pilotRngSeeds": [11]},
    {**GOOD_REPORT, "rngSeeds": []},
    {**GOOD_REPORT, "rngSeeds": 11},
    {**GOOD_REPORT, "rngSeeds": [11, "12"]},
    {**GOOD_REPORT, "rngSeeds": [11, True]},
    {**GOOD_REPORT, "rngSeeds": [11.0]},
])
def test_full_refuses_rng_seeds_that_are_absent_or_not_a_non_empty_int_list(
        tmp_path: Path, report: dict[str, Any]) -> None:
    refused_full(tmp_path, "rngSeeds", report)


@pytest.mark.parametrize("ref", [None, "", "  ", 12, True])
def test_a_recheck_failed_gate_needs_a_stakeholder_ref_string(tmp_path: Path, ref: Any) -> None:
    decision = {**PROCEED} if ref is None else {**PROCEED, "stakeholderRef": ref}
    refused_full(tmp_path, "stakeholderRef", {**GOOD_REPORT, "stakeholderDecision": decision,
                                              "missingCueCheck": {"state": "recheck-failed"}})


@pytest.mark.parametrize("report, match", [
    ({**GOOD_REPORT, "stakeholderDecision": {"cap": 25}}, "None, not 'proceed'"),
    ({**GOOD_REPORT, "missingCueCheck": {}}, "state None"),
    ({**GOOD_REPORT, "missingCueCheck": "passed"}, "state None"),
    ("{not json", "not JSON"),
    ([GOOD_REPORT], "not a JSON object"),
])
def test_full_refuses_absent_gate_keys_and_a_malformed_report(
        tmp_path: Path, report: Any, match: str) -> None:
    refused_full(tmp_path, match, report)


def test_full_needs_a_pilot_report(tmp_path: Path) -> None:
    refused_init(tmp_path / "camp", "no guard pilot report", phase="full",
                 data_dir=tmp_path / "data")


@pytest.mark.parametrize("point, previous", [("message-type-classifier", "guard-judge"),
                                             ("join-in-judge", "message-type-classifier")])
def test_full_needs_the_previous_points_campaign_finalized(
        tmp_path: Path, point: str, previous: str) -> None:
    data = tmp_path / "data"
    pilot_report(data, stakeholderDecision=PROCEED, missingCueCheck={"state": "passed"},
                 rngSeeds=[11])
    refused_init(tmp_path / "camp", f"{previous}'s campaign is not finalized", point=point,
                 phase="full", data_dir=data)
    (data / previous / "v1").mkdir(parents=True)
    (data / previous / "v1" / "generation.json").write_text("{}", "utf-8")
    Campaign(tmp_path / "camp", clock=Clock()).init(point=point, phase="full", data_dir=data)


# --- plan5 §3.2: seed identity -------------------------------------------------------------------


def test_init_full_loads_the_pilot_rng_seeds_as_held(tmp_path: Path) -> None:
    data = tmp_path / "data"
    pilot_report(data, stakeholderDecision=PROCEED, missingCueCheck={"state": "passed"},
                 rngSeeds=[101, 202])
    camp = Campaign(tmp_path / "camp", clock=Clock())
    camp.init(point="guard-judge", phase="full", data_dir=data)
    assert camp.state().held_rng_seeds == {101, 202}
    before = ledger(camp)
    with pytest.raises(CampaignRefused, match="RNG seed 202"):
        camp.record_draw(seeds.draw_seeds("guard-judge", 2, 202, design_stratum=None))
    assert ledger(camp) == before


def test_a_held_rng_seed_is_refused_and_nothing_is_written(tmp_path: Path) -> None:
    camp = smoke(tmp_path)
    camp.record_draw(seeds.draw_seeds("guard-judge", 3, 7, design_stratum=None))
    before = ledger(camp)
    with pytest.raises(CampaignRefused, match="RNG seed 7 is already held"):
        camp.record_draw(seeds.draw_seeds("guard-judge", 2, 7, design_stratum="turns"))
    assert ledger(camp) == before


def test_a_fresh_rng_seed_with_a_held_seed_id_is_refused(tmp_path: Path) -> None:
    camp = smoke(tmp_path)
    first = seeds.draw_seeds("guard-judge", 3, 7, design_stratum=None)
    camp.record_draw(first)
    fresh = seeds.draw_seeds("guard-judge", 2, 8, design_stratum=None)
    clashing = seeds.SeedDraw(role=fresh.role, rng_seed=fresh.rng_seed, count=fresh.count,
                              design_stratum=None,
                              seeds=(fresh.seeds[0], first.seeds[1]))
    before = ledger(camp)
    with pytest.raises(CampaignRefused, match=f"seedId {first.seeds[1].seed_id}"):
        camp.record_draw(clashing)
    assert ledger(camp) == before


def test_a_recorded_draw_holds_its_seeds_and_rng_seed(tmp_path: Path) -> None:
    camp = smoke(tmp_path)
    draw = seeds.draw_seeds("guard-judge", 3, 7, design_stratum=None)
    camp.record_draw(draw)
    state = camp.state()
    assert 7 in state.held_rng_seeds
    assert set(state.seeds) == {seed.seed_id for seed in draw.seeds}
    (event,) = [e for e in camp.events() if e["event"] == "seed-draw"]
    assert event["draw"] == draw.to_record()


def test_fresh_rng_seed_redraws_a_held_value(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = smoke(tmp_path)
    camp.record_draw(seeds.draw_seeds("guard-judge", 2, 41, design_stratum=None))
    values = iter([41, 42])
    monkeypatch.setattr(campaign.secrets, "randbits", lambda bits: next(values))
    assert camp.fresh_rng_seed() == 42


# --- resolve -------------------------------------------------------------------------------------


def safeguard_halt(camp: Campaign) -> None:
    run_batch(camp, "refusal")
    assert outcomes(camp)[-1]["reason"] == "safeguardStop"


def test_skip_batch_drops_a_label_batchs_items_and_reissues_nothing(tmp_path: Path) -> None:
    camp = smoke(tmp_path, items=3)
    safeguard_halt(camp)
    camp.resolve(decision="skip the flagged batch", skip_batch=True)
    state = camp.state()
    assert state.halt is None and state.batches["bat_0001"].dropped == "safeguardSkipped"
    assert state.dropped()["safeguardSkipped"] == {"seeds": 0, "items": 3}
    issued = len([e for e in camp.events() if e["event"] == "issued"])
    assert camp.next() == Stop("done", "nothing is pending")
    assert len([e for e in camp.events() if e["event"] == "issued"]) == issued


def test_skip_batch_drops_a_generation_batchs_seeds(tmp_path: Path) -> None:
    camp = smoke(tmp_path)
    run_batch(camp)
    draw = seeds.draw_seeds("guard-judge", 4, 9, design_stratum=None)
    camp.record_draw(draw)
    camp.queue_batch("generate", [seed.seed_id for seed in draw.seeds])
    safeguard_halt(camp)
    camp.resolve(decision="skip", skip_batch=True)
    assert camp.state().dropped()["safeguardSkipped"] == {"seeds": 4, "items": 0}


def test_skip_batch_is_for_a_safeguard_stop(tmp_path: Path) -> None:
    camp = smoke(tmp_path)
    run_batch(camp, "opus-driver")
    before = ledger(camp)
    with pytest.raises(CampaignRefused, match="safeguardStop"):
        camp.resolve(decision="skip", skip_batch=True)
    assert ledger(camp) == before


def test_retry_reissues_the_halted_batch_under_a_new_name_with_no_attempt(
        tmp_path: Path) -> None:
    camp = smoke(tmp_path)
    safeguard_halt(camp)
    halted = outcomes(camp)[-1]["opaque"]
    camp.resolve(decision="retry")
    resolved = camp.events()[-1]
    assert {k: resolved[k] for k in ("event", "decision", "batch", "reason", "action")} == {
        "event": "resolved", "decision": "retry", "batch": "bat_0001",
        "reason": "safeguardStop", "action": "retry"}
    action = serve(camp)
    assert action.batch == "bat_0001" and action.opaque != halted
    assert camp.state().batches["bat_0001"].counted == 0


def test_resolve_refuses_when_not_halted_or_with_two_actions(tmp_path: Path) -> None:
    camp = smoke(tmp_path)
    with pytest.raises(CampaignRefused, match="not halted"):
        camp.resolve(decision="x")
    run_batch(camp, "opus-driver")
    before = ledger(camp)
    with pytest.raises(CampaignRefused, match="at most one"):
        camp.resolve(decision="x", skip_batch=True, repin_instructions=True)
    with pytest.raises(CampaignRefused, match="needs a text"):
        camp.resolve(decision="  ")
    assert ledger(camp) == before


def test_repin_model_writes_the_restart_event_and_the_new_pin(tmp_path: Path) -> None:
    camp = smoke(tmp_path)
    safeguard_halt(camp)
    camp.resolve(decision="move generation", repin_model=("generator", "claude-opus-4-8"))
    event = camp.events()[-1]
    assert (event["event"], event["model"]) == ("generation-restart", "claude-opus-4-8")
    assert camp.state().pins["generator"] == "claude-opus-4-8"
    assert camp.state().halt is None


def test_repin_model_refuses_the_other_roles_model(tmp_path: Path) -> None:
    camp = smoke(tmp_path)
    safeguard_halt(camp)
    with pytest.raises(CampaignRefused, match="equals the labeler's"):
        camp.resolve(decision="x", repin_model=("generator", "claude-sonnet-5-5"))
    camp.resolve(decision="x", repin_model=("labeler", "claude-sonnet-5-6"))
    assert camp.events()[-1]["event"] == "labelling-restart"


def test_repin_instructions_declares_the_halted_runs_set(tmp_path: Path) -> None:
    camp = Campaign(tmp_path / "camp", claude_dir=tmp_path / "claude", clock=Clock())
    camp.init(point="guard-judge", phase="smoke",
              fixture=write_fixture(tmp_path / "f.jsonl", 2))
    roles = {role: {**spec, "instructions": [["CLAUDE.md", "0" * 64]]}
             for role, spec in ROLES.items()}
    camp.record_expectations(guard_command=GUARD_COMMAND, roles=roles)
    run_batch(camp)
    assert outcomes(camp)[-1]["reason"] == "instructionsMismatch"
    camp.resolve(decision="the instructions set is empty, as declared", repin_instructions=True)
    assert camp.state().instructions["labeler"] == []
    run_batch(camp)
    assert outcomes(camp)[-1]["outcome"] == "ingested"


def test_resolve_refuses_while_the_lock_is_held(tmp_path: Path) -> None:
    import fcntl
    camp = smoke(tmp_path)
    run_batch(camp, "opus-driver")
    before = ledger(camp)
    with open(camp.store / "driver.lock", "a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        with pytest.raises(CampaignRefused, match="locked"):
            camp.resolve(decision="x")
    assert ledger(camp) == before


# --- status --------------------------------------------------------------------------------------


def test_status_reports_halt_pause_refusal_and_drops(tmp_path: Path) -> None:
    camp = smoke(tmp_path)
    safeguard_halt(camp)
    camp.pause_path.touch()
    (camp.store / "refusal").write_text("2026-11-02T09:00:00Z hooks key present\n", "utf-8")
    text = "\n".join(camp.status())
    assert "HALTED: safeguardStop" in text and "PAUSED" in text
    assert "REFUSED: 2026-11-02T09:00:00Z hooks key present" in text
    camp.resolve(decision="skip", skip_batch=True)
    assert "safeguardSkipped" in "\n".join(camp.status())


def test_status_names_the_last_issued_session(tmp_path: Path) -> None:
    """plan6 §6: an orphan is found by the last `issued` `sessionId`."""
    camp = smoke(tmp_path)
    assert not any(line.startswith("last issued") for line in camp.status())
    run_batch(camp, "no-tool-use")
    action = serve(camp)
    camp.issue(action, session_id="0f0f0f0f-0000-4000-8000-0000000000aa")
    assert "last issued: bat_0001 session 0f0f0f0f-0000-4000-8000-0000000000aa" in camp.status()


@pytest.mark.parametrize("command", [
    lambda camp: camp.status(), lambda camp: camp.next(), lambda camp: camp.ingest("x" * 16),
    lambda camp: camp.resolve(decision="x"),
    lambda camp: camp.record_expectations(guard_command=GUARD_COMMAND, roles=ROLES),
])
def test_a_torn_last_ledger_line_refuses_every_command_naming_its_offset(
        tmp_path: Path, command: Any) -> None:
    """P10-n2: a crash mid-append leaves a line with no newline; it may be a commit point, so it
    is refused, never dropped, and nothing is appended after it."""
    camp = smoke(tmp_path)
    offset = len(ledger(camp))
    with open(camp.ledger_path, "ab") as handle:
        handle.write(b'{"event":"queued","at":"2026')
    before = ledger(camp)
    with pytest.raises(CampaignRefused, match=f"last line, at byte {offset}, is torn"):
        command(camp)
    assert ledger(camp) == before


def test_a_ledger_line_that_is_not_an_event_is_refused_naming_its_offset(tmp_path: Path) -> None:
    camp = smoke(tmp_path)
    good = ledger(camp)
    for bad, match in ((b"{oops\n", "not JSON"), (b"[1]\n", "not an event")):
        camp.ledger_path.write_bytes(good + bad)
        with pytest.raises(CampaignRefused, match=f"line at byte {len(good)} is {match}"):
            camp.events()


def test_an_event_text_with_a_unicode_line_separator_reads_back(tmp_path: Path) -> None:
    """The ledger is split on `\\n` only: `ensure_ascii=False` keeps U+2028 raw."""
    camp = smoke(tmp_path)
    run_batch(camp, "opus-driver")
    camp.resolve(decision="line one\u2028line two")
    assert camp.events()[-1]["decision"] == "line one\u2028line two"


def test_parse_duration() -> None:
    assert campaign.parse_duration("45m") == timedelta(minutes=45)
    assert campaign.parse_duration("6h") == timedelta(hours=6)
    assert campaign.parse_duration("90") == timedelta(seconds=90)
    for bad in ("", "0h", "-1h", "1.5h", "h"):
        with pytest.raises(ValueError):
            campaign.parse_duration(bad)


def test_the_transcript_slug_replaces_every_non_alphanumeric_character() -> None:
    assert campaign.transcript_slug(Path("/tmp/a_b.c/exchange")) == "-tmp-a-b-c-exchange"
    assert campaign.transcript_slug(Path("/tmp/Ação 9/ex")) == "-tmp-A--o-9-ex"   # ASCII only
