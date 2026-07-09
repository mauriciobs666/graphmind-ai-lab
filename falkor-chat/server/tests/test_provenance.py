"""Integration tests for the K-013 `EMITTED` agent-answer provenance (repo layer).

`post_agent_answer` / `read_provenance` / `read_citing_answers` map 1:1 to
`docs/QUERIES.md` §10 (verified in `docs/plans/m2-agent-participant.md`). Tests
run against the live `ws:test` graph (conftest bootstraps schema + wipes per
test), mirroring `test_graphrag.py`/`test_repository.py`.

The `EMITTED` edge is `(answer:Message)-[:EMITTED {score, rank}]->(seed:Message)`,
written **inside the same GRAPH.QUERY** as the answer's §4 write (atomicity).
"""

from __future__ import annotations

WS = "test"


def _seed_thread(repo, *, channel_id="c1", thread_id="t1", seeds=None):
    """Create channel→thread + user u1 + agent bot1, post the seed messages.

    `seeds` = [(msg_id, text)]; the first is a first-message write, the rest
    subsequent. Returns nothing — the graph is seeded in place.
    """
    seeds = seeds or []
    repo.ensure_user(WS, user_id="u1")
    repo.ensure_agent(WS, agent_id="bot1", name="Bot")
    repo.create_channel(WS, channel_id=channel_id, name=channel_id, created_at=1)
    repo.create_thread(
        WS, channel_id=channel_id, thread_id=thread_id, title="t", created_at=1
    )
    first = True
    ts = 10
    for msg_id, text in seeds:
        write = repo.post_first_message if first else repo.post_subsequent_message
        write(
            WS, thread_id=thread_id, msg_id=msg_id, author_id="u1",
            text=text, role="user", created_at=ts,
        )
        first = False
        ts += 1


# ── post_agent_answer: writes the answer + EMITTED edges ──────────────────────


def test_post_agent_answer_writes_agent_message_and_provenance(repo):
    _seed_thread(repo, seeds=[("s1", "seed one"), ("s2", "seed two"), ("s3", "seed three")])

    st = repo.post_agent_answer(
        WS, thread_id="t1", msg_id="ag1", author_id="bot1",
        text="the answer", role="assistant", created_at=100,
        seeds=[("s1", 0.0), ("s2", 0.006), ("s3", 0.5)],
    )

    assert st is not None and st.written and st.author_found

    prov = repo.read_provenance(WS, msg_id="ag1")
    assert [p["seedMsgId"] for p in prov] == ["s1", "s2", "s3"]  # ordered by rank
    assert [p["rank"] for p in prov] == [0, 1, 2]
    assert [p["score"] for p in prov] == [0.0, 0.006, 0.5]
    assert [p["role"] for p in prov] == ["user", "user", "user"]
    assert prov[0]["text"] == "seed one"


def test_post_agent_answer_role_derived_assistant_and_author_is_agent(repo):
    _seed_thread(repo, seeds=[("s1", "seed one")])
    repo.post_agent_answer(
        WS, thread_id="t1", msg_id="ag1", author_id="bot1",
        text="the answer", role="assistant", created_at=100,
        seeds=[("s1", 0.1)],
    )

    msg = repo.get_message(WS, msg_id="ag1")
    assert msg["role"] == "assistant"
    assert msg["authorId"] == "bot1"
    assert msg["authorType"] == ["Agent"]  # K-007 authorship invariant


def test_post_agent_answer_drops_unknown_seeds(repo):
    _seed_thread(repo, seeds=[("s1", "seed one"), ("s2", "seed two")])
    repo.post_agent_answer(
        WS, thread_id="t1", msg_id="ag1", author_id="bot1",
        text="answer", role="assistant", created_at=100,
        seeds=[("s1", 0.1), ("zz", 0.2), ("s2", 0.3)],  # zz unknown
    )
    prov = repo.read_provenance(WS, msg_id="ag1")
    assert sorted(p["seedMsgId"] for p in prov) == ["s1", "s2"]  # zz dropped


def test_post_agent_answer_empty_seeds_commits_zero_edges(repo):
    _seed_thread(repo, seeds=[("s1", "seed one")])
    st = repo.post_agent_answer(
        WS, thread_id="t1", msg_id="ag1", author_id="bot1",
        text="answer", role="assistant", created_at=100, seeds=[],
    )
    assert st.written
    assert repo.read_provenance(WS, msg_id="ag1") == []
    # the message itself is committed & readable (verified no-op guard)
    assert repo.get_message(WS, msg_id="ag1")["text"] == "answer"


def test_post_agent_answer_none_seeds_commits_zero_edges(repo):
    _seed_thread(repo, seeds=[("s1", "seed one")])
    st = repo.post_agent_answer(
        WS, thread_id="t1", msg_id="ag1", author_id="bot1",
        text="answer", role="assistant", created_at=100, seeds=None,
    )
    assert st.written
    assert repo.read_provenance(WS, msg_id="ag1") == []


def test_post_agent_answer_dup_replay_leaves_provenance_exactly_once(repo):
    _seed_thread(repo, seeds=[("s1", "seed one"), ("s2", "seed two")])
    args = dict(
        thread_id="t1", msg_id="ag1", author_id="bot1", text="answer",
        role="assistant", created_at=100, seeds=[("s1", 0.1), ("s2", 0.2)],
    )
    first = repo.post_agent_answer(WS, **args)
    assert first.written and not first.dup_msg
    assert len(repo.read_provenance(WS, msg_id="ag1")) == 2

    replay = repo.post_agent_answer(WS, **args)
    assert replay.dup_msg and not replay.written
    # exactly-once: still 2 edges, not 4
    assert len(repo.read_provenance(WS, msg_id="ag1")) == 2


def test_post_agent_answer_returns_none_when_no_tail(repo):
    # subsequent-path write with no TAIL → anchor miss → None (service retries first)
    repo.ensure_agent(WS, agent_id="bot1", name="Bot")
    repo.create_channel(WS, channel_id="c1", name="c1", created_at=1)
    repo.create_thread(WS, channel_id="c1", thread_id="t1", title="t", created_at=1)
    st = repo.post_agent_answer(
        WS, thread_id="t1", msg_id="ag1", author_id="bot1",
        text="answer", role="assistant", created_at=100, seeds=[],
    )
    assert st is None


def test_post_agent_answer_first_writes_head_and_provenance(repo):
    # First-path variant: agent answers into a headless thread (defensive fallback).
    repo.ensure_user(WS, user_id="u1")
    repo.ensure_agent(WS, agent_id="bot1", name="Bot")
    repo.create_channel(WS, channel_id="c1", name="c1", created_at=1)
    repo.create_thread(WS, channel_id="c1", thread_id="t1", title="t", created_at=1)
    # seed messages must exist to be cited — put them in another thread
    repo.create_thread(WS, channel_id="c1", thread_id="t0", title="t0", created_at=1)
    repo.post_first_message(
        WS, thread_id="t0", msg_id="s1", author_id="u1",
        text="seed one", role="user", created_at=10,
    )

    st = repo.post_agent_answer_first(
        WS, thread_id="t1", msg_id="ag1", author_id="bot1",
        text="answer", role="assistant", created_at=100, seeds=[("s1", 0.1)],
    )
    assert st is not None and st.written and not st.had_head
    prov = repo.read_provenance(WS, msg_id="ag1")
    assert [p["seedMsgId"] for p in prov] == ["s1"]


# ── read_citing_answers: reverse (impact) read ────────────────────────────────


def test_read_citing_answers_reverse(repo):
    _seed_thread(repo, seeds=[("s1", "seed one"), ("s2", "seed two")])
    repo.post_agent_answer(
        WS, thread_id="t1", msg_id="ag1", author_id="bot1",
        text="answer", role="assistant", created_at=100,
        seeds=[("s1", 0.1), ("s2", 0.2)],
    )
    citing = repo.read_citing_answers(WS, seed_msg_id="s1")
    assert [c["answerMsgId"] for c in citing] == ["ag1"]
    assert citing[0]["role"] == "assistant"
    assert citing[0]["score"] == 0.1
    assert citing[0]["rank"] == 0


def test_read_provenance_empty_for_message_without_seeds(repo):
    _seed_thread(repo, seeds=[("s1", "seed one")])
    assert repo.read_provenance(WS, msg_id="s1") == []
    assert repo.read_citing_answers(WS, seed_msg_id="s1") == []
