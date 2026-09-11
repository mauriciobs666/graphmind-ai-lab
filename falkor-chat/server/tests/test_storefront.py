"""Storefront core — participant registry, join, token verification, turn map.

`docs/plans/salesperson-ui.md` S6. **Integration tests against the live `ws:test`
graph**, not against a fake repository, and deliberately so: every property this
step has to hold is a property of *the graph being the registry*, and a fake
repository is exactly the thing that cannot tell you whether that is true.

Two of the done-conditions are in the danger zone the rest of this build has been
bitten by — evidence that stays green while asserting nothing:

- *"wrong, absent, malformed and deleted-participant tokens all resolve to
  `None`"* would pass in full against a `resolve_token` that returned `None`
  unconditionally. Every negative case here is therefore paired with the
  positive control in `test_a_valid_token_resolves_to_that_participant`, and the
  negatives that matter most (a deleted participant, another participant's
  token) are written as *transitions* — the same credential resolving before and
  not after — so "always `None`" cannot pass them.
- *"a `Storefront` rebuilt from scratch resolves a token minted by the previous
  instance"* passes trivially if the record map is process-global or a fixture
  leaks it. `test_a_rebuilt_storefront_resolves_a_token_minted_by_the_previous_instance`
  builds the second `Storefront` on its own `Repository` over its own connection
  and asserts its cache is empty before it answers, so the only thing the two
  instances share is the graph.

Both were mutation-tested (see the docstrings on those two tests for the exact
mutation each one catches).
"""

from __future__ import annotations

import hashlib
import hmac
import inspect
import logging
import re
import threading
import time
from concurrent.futures import Future
from pathlib import Path

import pytest
from redis import exceptions as redis_exceptions

from falkorchat import config, db, storefront
from falkorchat.repository import Repository
from falkorchat.services import Services
from falkorchat.storefront import (
    IDLE_TURN,
    TURN_IDLE,
    TURN_QUEUED,
    TURN_THINKING,
    DemoNotSeededError,
    OrderTransitionRefusedError,
    ParticipantRecord,
    QuiesceTimeoutError,
    ResetStateUnknownError,
    Storefront,
    StorefrontError,
    TurnBooking,
    TurnState,
    UnknownOrderError,
    UnknownParticipantError,
    UnscopedParticipantError,
    hash_token,
    parse_bearer,
)

WS = "test"
AGENT = "assistant"
LOCALES = ("en", "pt-BR", "es")

# How long the stub turn in `test_the_reset_waits_for_an_in_flight_turn…` holds
# the turn map before finishing. Nothing asserts a duration against it, so it is
# not a timing budget; it only has to be long enough that the reset is reliably
# issued *while* the turn is still in flight. Since S8-1 that is a margin, not
# nothing — `started_at` is stamped on `_call_bounded`'s worker thread, so this
# has to outlast that thread's start skew, measured at 0.12–0.15 ms over 20
# samples against the 150 ms here. Losing it fails loudly and in the right
# words ("the reset was not issued while the turn was in flight"), which is the
# whole difference S8-1 bought.
TURN_WORK_S = 0.15

# The ceiling `_call_bounded` gives a `quiesce_s=0` reset. Not a performance
# assertion — it is what turns a broken `_await_quiesce` deadline into a
# *failing test* instead of a hang (review
# `docs/reviews/salesperson-ui-impl.md` `## Pass 7`, S7-3).
#
# The three calls that take this bound are not the same size. Measured over
# three runs: ~0.2 ms twice, for the two refusals, which come back before they
# touch the graph — and ~2.5 ms once, in
# `test_an_idle_participant_is_not_made_to_wait`, the only bounded call that
# commits a write. So the margin is ~360x, not the ~5000x the refusals alone
# suggest, and that write is where tightening this constant bites first: a
# tripped bound there leaves a daemon thread that goes on to delete and re-mint
# a subgraph in the shared `ws:test` *after* its test has ended, under a message
# blaming a hung wait (Pass 8, S8-2).
#
# It stays at 1.0 s rather than being widened away from that write, because the
# tightness is doing work: with `_await_quiesce` mutated to sleep a blind 2 s
# and then report idle, this bound is what reddens the idle test — the one
# assertion in the file that answers "the reset waited when it had nothing to
# wait for".
IMMEDIATE_S = 1.0


def _call_bounded(fn, *args, seconds=IMMEDIATE_S, **kwargs):
    """Call `fn` on a daemon thread and **fail** if it has not returned in
    `seconds`.

    Returns `{started_at, returned_at, result}`, or **re-raises** whatever the
    call raised. The two instants are monotonic readings either side of the
    call, for tests that assert an *ordering* against another thread, and both
    are taken **on the thread that makes the call** — a `started_at` read on the
    calling thread instead would be the moment this test *asked* for the call,
    which precedes the moment the daemon thread begins it by the whole
    thread-start skew, and `started_at < …` would then tolerate that skew in the
    direction that passes (review `## Pass 8`, S8-1).

    Re-raising rather than handing back a captured exception is what keeps this
    from inverting `pytest.raises`: a call that raises fails the test whether or
    not the call site remembered to look, so the two sites that *expect* a
    refusal say so in the ordinary idiom — `with pytest.raises(...)` around the
    bounded call — and a site that expects success needs no bookkeeping
    assertion at all (S8-3).

    Why this and not the simpler `t0`/`assert elapsed < …` around a direct call:
    the thing being bounded is `_await_quiesce`'s own deadline, so a test that
    calls the reset inline inherits whatever budget the code under test
    computes. When that arithmetic is wrong the call does not come back slowly,
    it does not come back — and an assertion placed after it is never reached.
    Measured: with `deadline` extended by an hour, the elapsed-assert form still
    had to be killed at 30 s, exactly as the review's own run was killed at 25 s.

    There is no `pytest-timeout` in this venv (and installing one is `devops`'s
    call, not this test's), so the bound has to be one the test owns. The thread
    is a daemon so a genuinely hung call cannot keep the interpreter alive.
    """
    box: dict = {}

    def run():
        box["started_at"] = time.monotonic()
        try:
            box["result"] = fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 — re-raised on the caller below
            box["error"] = exc
        finally:
            box["returned_at"] = time.monotonic()

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout=seconds)
    assert not worker.is_alive(), (
        f"{getattr(fn, '__name__', fn)!r} did not return within {seconds}s — "
        "a wait bounded by the code under test hung instead of failing"
    )
    if "error" in box:
        raise box["error"]
    return box


def _probe(conn, cypher: str, params: dict | None = None):
    """Read-only probe of `ws:test` — assertions about what actually landed."""
    return db.workspace_graph(conn, WS).ro_query(cypher, params or {}).result_set


def _one(conn, cypher: str, params: dict | None = None):
    rows = _probe(conn, cypher, params)
    return rows[0][0] if rows else None


def _storefront(services, **overrides) -> Storefront:
    """A `Storefront` wired the way `create_app` will wire it (S8), but pinned to
    `ws:test` and to this suite's demo agent."""
    kwargs = {
        "presenter_key": "presenter-secret",
        "turn_workers": 2,
        "quiesce_s": 5.0,
        "ws": WS,
        "agent_id": AGENT,
        "locales": LOCALES,
    }
    kwargs.update(overrides)
    return Storefront(services, **kwargs)


@pytest.fixture()
def services(repo) -> Services:
    return Services(repo)


@pytest.fixture()
def seeded(repo, services):
    """A workspace with the demo `Agent` registered — the third precondition
    `ensure_participant` checks (graph note §3)."""
    repo.ensure_agent(WS, agent_id=AGENT, name="Demo agent", created_at=90)
    return _storefront(services)


def _bearer(record: ParticipantRecord) -> str:
    return f"Bearer {record.participant_id}.{record.token}"


# ── join: provisioning (§4.3) + the profile name (§4.10) ─────────────────────


def test_join_provisions_the_participant_subgraph_atomically(seeded, conn):
    record = seeded.join("Ada", "pt-BR")

    assert record.participant_id.startswith("p-")
    assert record.language == "pt-BR"
    assert record.display_name == "Ada"
    assert record.channel_id == f"ch-{record.participant_id}"
    assert record.thread_id == f"th-{record.participant_id}"

    pid = record.participant_id
    assert _one(
        conn, "MATCH (u:User {userId: $pid}) RETURN u.displayName", {"pid": pid}
    ) == "Ada"
    assert _one(
        conn, "MATCH (u:User {userId: $pid}) RETURN u.language", {"pid": pid}
    ) == "pt-BR"
    # The provenance marker both resets scope on — `ensure_participant` is its
    # only writer tree-wide, so without it neither reset ever resolves anyone.
    assert _one(
        conn, "MATCH (c:Channel {channelId: $cid}) RETURN c.participantId",
        {"cid": record.channel_id},
    ) == pid
    assert _one(
        conn,
        "MATCH (:Channel {channelId: $cid})-[:HAS_THREAD]->(t) RETURN t.threadId",
        {"cid": record.channel_id},
    ) == record.thread_id
    members = _probe(
        conn,
        "MATCH (mem)-[r:MEMBER_OF]->(:Channel {channelId: $cid}) "
        "RETURN coalesce(mem.userId, mem.agentId) AS id, r.role ORDER BY id",
        {"cid": record.channel_id},
    )
    assert members == [[AGENT, "assistant"], [pid, "member"]]


def test_join_writes_the_display_name_into_the_profile(seeded, conn):
    """§4.10: the name reaches the profile immediately, and the `Customer`
    anchor exists from the first moment — not thirty seconds later when the
    model gets round to asking for a name the participant already typed."""
    record = seeded.join("Ada", "en")

    profile = seeded._services.get_profile(seeded.context_for(record.participant_id))
    assert profile == {"name": "Ada", "deliveryAddress": None}
    assert _one(
        conn, "MATCH (c:Customer {customerId: $pid}) RETURN c.name",
        {"pid": record.participant_id},
    ) == "Ada"


def test_join_stores_only_the_hash_of_the_token(seeded, conn):
    record = seeded.join("Ada", "en")

    stored = _one(
        conn, "MATCH (u:User {userId: $pid}) RETURN u.tokenHash",
        {"pid": record.participant_id},
    )
    assert record.token is not None
    assert stored == hashlib.sha256(record.token.encode()).hexdigest()
    assert stored == hash_token(record.token)
    # The raw token is nowhere in the graph, under any property. The same scan
    # run against a value that *is* stored is the positive control — without it
    # a `keys()`/`n[k]` regression would turn this into a silent pass.
    scan = "MATCH (n) WHERE any(k IN keys(n) WHERE n[k] = $v) RETURN count(n)"
    assert _one(conn, scan, {"v": "Ada"}) >= 1
    assert _one(conn, scan, {"v": record.token}) == 0


def test_join_gives_each_participant_their_own_scope_and_credential(seeded):
    ada = seeded.join("Ada", "en")
    bob = seeded.join("Bob", "es")

    assert ada.participant_id != bob.participant_id
    assert ada.token != bob.token
    assert ada.channel_id != bob.channel_id
    assert ada.thread_id != bob.thread_id
    assert seeded.resolve_token(_bearer(ada)).participant_id == ada.participant_id
    assert seeded.resolve_token(_bearer(bob)).participant_id == bob.participant_id


def test_join_is_idempotent_when_the_participant_id_repeats(services, repo, conn):
    """Provisioning is idempotent: a replayed id writes no second `User`,
    `Channel` or `Thread`, and keeps the original `joinedAt`.

    Only reachable by pinning `id_gen` — in production the id is a server-minted
    `uuid4` no client can supply. The replay still returns a **working**
    credential, because `ensure_participant` is deliberately not a token-rotation
    path (it returns the stored ids and does not write the fresh hash), so `join`
    writes the new hash through `set_participant_record`.
    """
    repo.ensure_agent(WS, agent_id=AGENT, created_at=90)
    clock = iter([1000, 2000])
    shop = _storefront(
        services, id_gen=lambda: "p-fixed", clock=lambda: next(clock)
    )

    first = shop.join("Ada", "en")
    second = shop.join("Ada Renamed", "es")

    assert _one(conn, "MATCH (u:User) RETURN count(u)") == 1
    assert _one(conn, "MATCH (c:Channel) RETURN count(c)") == 1
    assert _one(conn, "MATCH (t:Thread) RETURN count(t)") == 1
    assert second.channel_id == first.channel_id
    assert second.thread_id == first.thread_id
    assert second.joined_at == 1000  # the original join, not the replay
    assert shop.resolve_token(_bearer(second)).display_name == "Ada Renamed"
    # The superseded credential is dead — the graph holds exactly one hash.
    assert shop.resolve_token(_bearer(first)) is None


def test_join_without_the_demo_agent_refuses_and_writes_nothing(services, conn):
    """Graph note §3 row 5: `agentMissing` means **nothing at all was written**.
    §4.9's readiness preflight should have caught this at boot; a participant
    seeing it means the deployment came up mis-seeded."""
    shop = _storefront(services)

    with pytest.raises(DemoNotSeededError) as exc:
        shop.join("Ada", "en")

    assert "seed_demo.sh" in str(exc.value)
    assert _one(conn, "MATCH (n) RETURN count(n)") == 0


# ── resolve_token: the positive control, then everything that must fail ──────


def test_a_valid_token_resolves_to_that_participant(seeded):
    """**The positive control.** Every `is None` assertion below is worthless
    without it: a `resolve_token` that returned `None` unconditionally passes
    the whole negative set."""
    record = seeded.join("Ada", "pt-BR")

    resolved = seeded.resolve_token(_bearer(record))

    assert resolved is not None
    assert resolved.participant_id == record.participant_id
    assert resolved.display_name == "Ada"
    assert resolved.language == "pt-BR"
    assert resolved.channel_id == record.channel_id
    assert resolved.thread_id == record.thread_id
    # A resolved record never carries the raw credential — only the mint path does.
    assert resolved.token is None


@pytest.mark.parametrize(
    "bearer",
    [
        pytest.param(None, id="absent"),
        pytest.param("", id="empty"),
        pytest.param("   ", id="whitespace"),
        pytest.param("Bearer", id="scheme-only"),
        pytest.param("Bearer ", id="scheme-no-credential"),
        pytest.param("Bearer no-separator-here", id="no-separator"),
        pytest.param("Bearer .just-a-token", id="empty-id-half"),
        pytest.param("Bearer p-abc.", id="empty-token-half"),
        pytest.param("Basic p-abc.tok", id="wrong-scheme"),
        pytest.param("Bearer presenter.some-presenter-token", id="presenter-credential"),
        pytest.param("Bearer p-unknown.whatever", id="unknown-participant"),
    ],
)
def test_absent_and_malformed_credentials_resolve_to_none(seeded, bearer):
    seeded.join("Ada", "en")  # a real participant exists, so `None` is a decision
    assert seeded.resolve_token(bearer) is None


def test_a_wrong_token_for_a_real_participant_resolves_to_none(seeded):
    record = seeded.join("Ada", "en")

    assert seeded.resolve_token(f"Bearer {record.participant_id}.wrong") is None
    # And a near-miss: the right token with one character changed.
    mangled = record.token[:-1] + ("A" if record.token[-1] != "A" else "B")
    assert seeded.resolve_token(f"Bearer {record.participant_id}.{mangled}") is None


def test_one_participants_token_never_resolves_under_anothers_id(seeded):
    """The isolation case: a credential is a *pair*, and neither half alone
    authenticates."""
    ada = seeded.join("Ada", "en")
    bob = seeded.join("Bob", "en")

    assert seeded.resolve_token(f"Bearer {ada.participant_id}.{bob.token}") is None
    assert seeded.resolve_token(f"Bearer {bob.participant_id}.{ada.token}") is None
    # Positive control on the same two credentials, so "always None" cannot pass.
    assert seeded.resolve_token(_bearer(ada)).participant_id == ada.participant_id
    assert seeded.resolve_token(_bearer(bob)).participant_id == bob.participant_id


def test_a_non_participant_user_id_resolves_to_none(seeded, repo):
    """`seed_demo.sh`'s `u1` and the lifespan's `config.USER_ID` node are `User`s
    with no `tokenHash`. They are not participants and no credential shape can
    make them one."""
    repo.ensure_user(WS, user_id="u1", display_name="Demo human")

    assert seeded.resolve_token("Bearer u1.anything") is None
    assert seeded.resolve_token(f"Bearer u1.{hash_token('anything')}") is None


def test_a_deleted_participant_stops_resolving_immediately(seeded, repo):
    """**Mutation-tested.** Give `resolve_token` a branch that returns a
    previously resolved record without re-reading the graph, and this test
    goes red: the deleted participant would keep authenticating out of stale
    memory. The graph read is the only thing that makes the reset real.
    """
    record = seeded.join("Ada", "en")
    assert seeded.resolve_token(_bearer(record)) is not None  # before

    repo.reset_all_participants(WS)

    assert seeded.resolve_token(_bearer(record)) is None  # after


def test_a_rebuilt_storefront_resolves_a_token_minted_by_the_previous_instance(
    seeded, services
):
    """Restart survival (§4.3) — **mutation-tested**. Make participant
    resolution stateful in-process (answer from an in-memory map instead of
    re-reading the graph) and this goes red: the second instance's map is
    empty, so every participant would be bounced to a fresh `participantId` and
    lose their cart and order, because `customerId == participantId`.

    The second `Storefront` is built on its own `Repository` over its own
    connection, so the only thing the two instances share is `ws:test` itself.
    """
    record = seeded.join("Ada", "pt-BR")
    bearer = _bearer(record)

    restarted = _storefront(Services(Repository(db.connect())))

    assert restarted is not seeded
    resolved = restarted.resolve_token(bearer)
    assert resolved is not None
    assert resolved.participant_id == record.participant_id
    assert resolved.display_name == "Ada"
    assert resolved.language == "pt-BR"
    assert resolved.channel_id == record.channel_id
    assert resolved.thread_id == record.thread_id


def test_resolve_token_reads_the_graph_on_every_call(seeded, repo):
    """A property change made behind the storefront's back is visible on the
    very next resolve — no restart, no cache invalidation call.

    This is the same invariant as the two mutation tests above, stated
    positively: if the cache answered, the language here would still be `en`.
    """
    record = seeded.join("Ada", "en")
    assert seeded.resolve_token(_bearer(record)).language == "en"

    repo.set_participant_record(WS, participant_id=record.participant_id, language="es")

    assert seeded.resolve_token(_bearer(record)).language == "es"


def test_the_token_comparison_goes_through_hmac_compare_digest(seeded, monkeypatch):
    """`hmac.compare_digest`, not `==` (§4.3), pinned **behaviourally**.

    Constant-time comparison has no observable behaviour — replacing the call
    with `!=` reddens no functional test in this file — so it has to be pinned by
    something other than the resolution outcome. A spy on the call is the
    strongest form available: it survives reformatting and renaming (unlike
    matching the call's source text), and unlike a source read it also goes red
    if a future branch *skips* the comparison entirely.

    What it does **not** claim: that the comparison is actually fast. That is a
    property of `hmac.compare_digest`, and this asserts only that we reach it.
    """
    record = seeded.join("Ada", "en")
    expected_hash = hash_token(record.token)
    calls: list[tuple] = []
    real = hmac.compare_digest

    def spy(a, b):
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(storefront.hmac, "compare_digest", spy)
    resolved = seeded.resolve_token(_bearer(record))

    assert resolved is not None  # the comparison was reached *and* succeeded
    # Exactly one call, and it compared the two hashes — never the raw token.
    assert calls == [(expected_hash, expected_hash)]
    assert record.token not in calls[0]


def test_resolve_token_never_compares_the_hash_with_an_operator():
    """The static half, narrowed to the clause that does the work.

    The spy above proves `compare_digest` is *reached*; it cannot prove nothing
    else compares the hash beside it. This reads `resolve_token`'s body for an
    equality operator — a tripwire against a future "simplify" that adds a
    short-circuit `if stored_hash == …` in front of the real call.

    Deliberately not matching the call's exact source text: that form reddens on
    a reformat or a local rename, i.e. on correct code (Pass 6, S6-4).
    """
    body = inspect.getsource(Storefront.resolve_token).split('"""', 2)[-1]

    assert "==" not in body
    assert "!=" not in body


# ── the registry cache — read-through, and never an auth path ────────────────


# ── parse_bearer ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Bearer p-abc.tok", ("p-abc", "tok")),
        ("bearer p-abc.tok", ("p-abc", "tok")),  # RFC 7235: scheme is case-insensitive
        ("  Bearer  p-abc.tok  ", ("p-abc", "tok")),
        ("p-abc.tok", ("p-abc", "tok")),  # the bare credential, without a scheme
        ("Bearer p-abc.tok.with.dots", ("p-abc", "tok.with.dots")),
    ],
)
def test_parse_bearer_accepts_the_credential_shapes_s8_will_hand_it(raw, expected):
    assert parse_bearer(raw) == expected


def test_a_minted_token_never_contains_the_separator(seeded):
    """`secrets.token_urlsafe` renders base64url — `[A-Za-z0-9_-]` — so a real
    token can never be split at the wrong dot. Asserted rather than assumed,
    because `parse_bearer` splits on the *first* separator."""
    tokens = [seeded.join(f"P{i}", "en").token for i in range(5)]

    for token in tokens:
        assert re.fullmatch(r"[A-Za-z0-9_-]+", token), token
        assert len(token) >= 40


# ── the turn-state map (§4.4 measure 1) ──────────────────────────────────────
#
# **Nothing here fabricates a map entry**, and it could not: `reserve_turn` is
# the only way one is created, every change to one names the booking that owns
# it, and there is no unconditional single-slot write left to fabricate with
# (§5.1's S9 row). So these tests drive the delivered protocol, which is also
# why they are the ones that catch a booking losing its slot.


def _pin_thinking(shop, participant_id: str) -> TurnBooking:
    """A **running** turn, reserved and flipped through the real protocol."""
    booking = shop.reserve_turn(participant_id)
    assert booking is not None
    assert shop.set_turn_state(participant_id, TURN_THINKING, booking=booking) is True
    return booking


def test_turn_state_defaults_to_idle(seeded):
    assert seeded.turn_state("p-nobody") == IDLE_TURN
    assert seeded.turn_payload("p-nobody") == {
        "state": TURN_IDLE, "queuePosition": 0,
        "lastTurn": None,
    }
    assert seeded.turn_in_flight("p-nobody") is False


def test_a_reservation_is_queued_first_in_line_and_gates_a_second_post(seeded):
    """`reserve_turn` is the `409` check and the booking as one step (§4.4
    measure 1a): the second reservation for the same participant **is** the
    refusal, and it books nothing.

    `queuePosition: 0` on a `queued` turn is ordinary and load-bearing — *first
    in line*, which S13 renders as a queue rather than as the absence of one
    (§5.2).
    """
    booking = seeded.reserve_turn("p-a")

    assert booking is not None
    assert seeded.turn_payload("p-a") == {
        "state": TURN_QUEUED, "queuePosition": 0,
        "lastTurn": None,
    }
    assert seeded.turn_in_flight("p-a") is True

    assert seeded.reserve_turn("p-a") is None
    assert seeded.turn_state("p-a").booking == booking

    assert seeded.set_turn_state("p-a", TURN_THINKING, booking=booking) is True
    assert seeded.turn_payload("p-a") == {
        "state": TURN_THINKING, "queuePosition": 0,
        "lastTurn": None,
    }
    assert seeded.turn_in_flight("p-a") is True


def test_the_queue_position_is_the_waiting_line_index_and_counts_down(seeded):
    """§5.2's definition, read at the **delivered default** rather than at
    `turn_workers=1`, which is the only setting the old number was right at.

    Four turns running and three waiting: the running ones occupy workers
    rather than places in line and report `0`, and the three waiting ones read
    `0/1/2` — so the fifth *arrival* is first in line, where a position taken
    at booking told it `4` (`docs/reviews/salesperson-ui-impl.md` `## Pass 17`,
    P17-2, reproduced). Then the head of the line starts running, and every
    number behind it **counts down** — the property a number fixed at booking
    cannot have at any `turn_workers`.
    """
    for i in range(4):
        _pin_thinking(seeded, f"p-run-{i}")
    waiting = [seeded.reserve_turn(f"p-wait-{i}") for i in range(3)]

    assert [seeded.turn_payload(f"p-run-{i}") for i in range(4)] == [
        {"state": TURN_THINKING, "queuePosition": 0, "lastTurn": None}
    ] * 4
    assert [
        seeded.turn_payload(f"p-wait-{i}")["queuePosition"] for i in range(3)
    ] == [0, 1, 2]

    assert seeded.set_turn_state(
        "p-wait-0", TURN_THINKING, booking=waiting[0]
    ) is True

    assert [
        seeded.turn_payload(f"p-wait-{i}")["queuePosition"] for i in range(3)
    ] == [0, 0, 1]


def test_a_running_turn_reports_zero_even_with_an_earlier_turn_still_queued(seeded):
    """§5.2's constant, stated over the **map** rather than over the executor:
    `queuePosition` is `0` whenever `state` is `thinking`, because a running
    turn has left the line — not because nothing happened to be booked before
    it.

    Without this the rule reads as an accident of ordering. Turns ordinarily
    start in booking order, so a `thinking` turn usually has no earlier
    *queued* one and a derivation that dropped the constant answers `0` anyway
    — measured: deleting the `state != TURN_QUEUED ⇒ 0` branch left the whole
    suite green. Here the earlier booking is still in the line, so the two
    readings disagree, and the one this pins is the contract: a participant
    whose turn is running must never be told they are waiting behind someone.
    """
    seeded.reserve_turn("p-first")
    later = seeded.reserve_turn("p-second")
    assert seeded.set_turn_state("p-second", TURN_THINKING, booking=later) is True

    assert seeded.turn_payload("p-second") == {
        "state": TURN_THINKING, "queuePosition": 0,
        "lastTurn": None,
    }
    assert seeded.turn_payload("p-first") == {
        "state": TURN_QUEUED, "queuePosition": 0,
        "lastTurn": None,
    }


def test_returning_to_idle_clears_the_entry(seeded):
    booking = _pin_thinking(seeded, "p-a")
    assert seeded.set_turn_state("p-a", TURN_IDLE, booking=booking) is True
    assert seeded.turn_state("p-a") == IDLE_TURN
    assert seeded.turn_in_flight("p-a") is False

    second = seeded.reserve_turn("p-a")
    assert seeded.release_turn("p-a", second) is True
    assert seeded.turn_in_flight("p-a") is False


def test_turn_state_is_per_participant(seeded):
    _pin_thinking(seeded, "p-a")
    seeded.reserve_turn("p-b")

    assert seeded.turn_in_flight("p-a") is True
    assert seeded.turn_in_flight("p-b") is True
    assert seeded.turn_in_flight("p-c") is False

    seeded.clear_all_turns()
    assert seeded.turn_in_flight("p-a") is False
    assert seeded.turn_in_flight("p-b") is False


def test_clear_all_turns_also_clears_every_dead_turn_latch(seeded):
    """§5.1's S9 row: reset-everyone deletes every participant's transcript,
    so a `lastTurn: "failed"` surviving it would point at a turn nobody can
    see any more — `clear_all_turns()` drops both maps in one call.
    """
    seeded._mark_turn_failed("p-a")  # noqa: SLF001 — stands in for a died turn
    seeded._mark_turn_failed("p-b")  # noqa: SLF001
    assert seeded.turn_payload("p-a")["lastTurn"] == "failed"
    assert seeded.turn_payload("p-b")["lastTurn"] == "failed"

    seeded.clear_all_turns()

    assert seeded.turn_payload("p-a")["lastTurn"] is None
    assert seeded.turn_payload("p-b")["lastTurn"] is None


def test_no_map_write_takes_effect_once_its_booking_has_lost_the_slot(seeded):
    """The **ownership** half of P17-1, at the unit the token protects.

    `clear_all_turns()` empties the map under turns that are still running —
    reset-all does exactly that (`docs/reviews/salesperson-ui-impl.md`
    `## Pass 17`, Ruling 2) — and a fresh accepted post then installs a
    *different* booking in that slot. Every write the displaced booking still
    holds must now be a no-op: the `thinking` flip and the release alike, the
    release included because it is the one path a worker-only rule does not
    cover (`## Pass 18`, P18-3).

    The positive controls are the point: an implementation that refused **every**
    write would pass the three `is False` lines and fail the two below them.
    """
    old = seeded.reserve_turn("p-a")
    seeded.clear_all_turns()
    new = seeded.reserve_turn("p-a")

    assert new is not None
    assert new != old

    assert seeded.set_turn_state("p-a", TURN_THINKING, booking=old) is False
    assert seeded.release_turn("p-a", old) is False
    assert seeded.turn_state("p-a") == TurnState(state=TURN_QUEUED, booking=new)
    assert seeded.turn_in_flight("p-a") is True

    assert seeded.set_turn_state("p-a", TURN_THINKING, booking=new) is True
    assert seeded.release_turn("p-a", new) is True
    assert seeded.turn_in_flight("p-a") is False


def test_the_arrival_ordinal_is_per_storefront_monotonic_and_never_reused(
    seeded, services
):
    """The ordinal does two jobs — it orders the line and it identifies the
    booking — and a collision corrupts both at once (§5.1's S9 row).

    Three spellings fail here, and each is a plausible reading of *arrival
    ordinal*. **`len(self._turns)`** and **a counter reset in
    `clear_all_turns()`** both restart, so the third assertion is `0 > 0`.
    **A module-level counter** satisfies monotonicity and fails the last one:
    every other piece of this object's mutable state is per-instance, for the
    reason `Storefront.__init__`'s docstring gives — a second `Storefront`
    shares nothing with the first but the graph, which is what makes the
    restart-survival test in this file mean anything.
    """
    first = [seeded.reserve_turn(f"p-{i}").ordinal for i in range(3)]
    assert first == [first[0], first[0] + 1, first[0] + 2]

    seeded.clear_all_turns()

    assert seeded.reserve_turn("p-0").ordinal > max(first)

    other = _storefront(services)
    assert other.reserve_turn("p-0").ordinal == 0


# ── the turn queue (§4.4 measure 1) ──────────────────────────────────────────
#
# The queue's *behaviour* — positions 0/1/2, completion order, the `409`
# ordering, the poll budget and the shutdown drain — is asserted over HTTP in
# `tests/test_storefront_api.py`, where the participants and the graph writes
# are real. What is left here is what only this layer can say: what the worker
# is handed, and what it does when the trigger fails.


class _RecordingTrigger:
    """The `WorkflowTrigger` seam: records the one call `_run_turn` makes."""

    def __init__(self, explode: bool = False) -> None:
        self.calls: list[dict] = []
        self.done = threading.Event()
        self._explode = explode

    def maybe_trigger(self, ctx, **kwargs):  # noqa: ANN001, ANN003
        self.calls.append({"ctx": ctx, **kwargs})
        try:
            if self._explode:
                raise RuntimeError("the LLM endpoint is down")
        finally:
            self.done.set()


def _drain(shop, future, trigger=None) -> None:
    """Wait for one enqueued turn, failing rather than hanging."""
    if trigger is not None:
        assert trigger.done.wait(timeout=IMMEDIATE_S), "the turn never ran"
    future.result(timeout=IMMEDIATE_S)


def _enqueue(shop, ctx, record, posted):
    """The request thread's whole sequence, minus the graph write: reserve the
    slot (the `409` check and the booking as one step), then submit.

    Spelled as a helper because it is a **sequence**, not a call: a test that
    submitted without reserving would be exercising a path no route can take.
    """
    booking = shop.reserve_turn(record.participant_id)
    assert booking is not None, "the participant already had a turn in flight"
    return shop.enqueue_turn(ctx, record, posted, booking)


def test_the_turn_worker_carries_the_participants_language_in_the_run_ctx(services):
    """§4.5's carrier, and §5.1's reason for handing the record in.

    `run_ctx={"language": …}` comes off the `ParticipantRecord` the *request
    thread* resolved, which is the whole argument for `enqueue_turn` taking one:
    the worker never has to answer "who is `p-…`" for itself. Everything else
    the trigger receives is the posted row, unchanged.
    """
    trigger = _RecordingTrigger()
    shop = _storefront(services, trigger=trigger)
    record = ParticipantRecord(
        participant_id="p-ada", display_name="Ada", language="pt-BR",
        channel_id="ch-ada", thread_id="th-ada", joined_at=1,
    )
    posted = {
        "msgId": "m-1", "threadId": "th-ada", "text": "olá",
        "role": "member", "mentions": [AGENT],
    }
    ctx = shop.context_for("p-ada")

    _drain(shop, _enqueue(shop, ctx, record, posted), trigger)

    assert len(trigger.calls) == 1
    call = trigger.calls[0]
    assert call["ctx"] is ctx
    assert call["run_ctx"] == {"language": "pt-BR"}
    assert call["thread_id"] == "th-ada"
    assert call["msg_id"] == "m-1"
    assert call["text"] == "olá"
    assert call["role"] == "member"
    assert call["mentions"] == [AGENT]


def test_the_turn_worker_never_resolves_a_participant_record(services, monkeypatch):
    """The other half of "the request thread hands the record in": the worker
    issues **no** participant read of its own.

    Pinned at the repository call rather than by reading `_run_turn`, so a
    worker that reached for a cache or re-queried the registry reddens here
    whichever way it spelled it.
    """
    trigger = _RecordingTrigger()
    shop = _storefront(services, trigger=trigger)
    reads: list[str] = []
    original = shop._repo.get_participant_record

    def counting(*args, **kwargs):
        reads.append(kwargs.get("participant_id"))
        return original(*args, **kwargs)

    monkeypatch.setattr(shop._repo, "get_participant_record", counting)
    record = ParticipantRecord(
        participant_id="p-ada", display_name="Ada", language="en",
        channel_id="ch-ada", thread_id="th-ada", joined_at=1,
    )
    posted = {"msgId": "m-1", "threadId": "th-ada", "text": "hi",
              "role": "member", "mentions": [AGENT]}

    _drain(shop, _enqueue(shop, shop.context_for("p-ada"), record, posted), trigger)

    assert reads == []
    # the positive control: the spy really is on the seam a resolution uses
    shop.resolve_token("Bearer p-ada.whatever")
    assert reads == ["p-ada"]


def test_a_turn_whose_trigger_raises_is_isolated_and_still_clears_the_gate(
    services, caplog
):
    """The turn runs after the `200` has been sent, so nothing it raises has a
    response to reach — it is logged and swallowed, exactly as
    `background._safe_run_workflow` does on both existing transports.

    What must **not** be swallowed with it is the map entry: a turn that died
    without clearing would leave that participant permanently `409`-refused,
    unable to retry the thing that failed. The dominant cause is ordinary — an
    LLM endpoint that is down, or the 180 s agent timeout.

    **And "logged" is asserted, not assumed.** Before `turn.lastTurn` (S9c)
    this record was the *only* evidence anywhere that a turn died — the
    participant sees their message, no reply, and a composer that quietly
    re-enables. Replacing the `_log.exception(...)` with `pass` left the whole
    suite green at **280 passed** (`docs/reviews/salesperson-ui-impl.md`
    `## Pass 17`, P17-4, mutation D), so the four assertions below are what
    make the swallow observable: one record, at `ERROR`, carrying the
    traceback, naming both the participant and the message the operator would
    need to find the turn.

    **The participant now has a second, participant-facing source of the same
    fact** — `turn.lastTurn == "failed"`, asserted below *after* the entry the
    exception happened on is gone (`turn_state` reads `IDLE_TURN`), which is
    exactly the ordering the latch exists to survive.
    """
    caplog.set_level(logging.DEBUG, logger="falkorchat.storefront")
    trigger = _RecordingTrigger(explode=True)
    shop = _storefront(services, trigger=trigger)
    record = ParticipantRecord(
        participant_id="p-ada", display_name="Ada", language="en",
        channel_id="ch-ada", thread_id="th-ada", joined_at=1,
    )
    posted = {"msgId": "m-1", "threadId": "th-ada", "text": "hi",
              "role": "member", "mentions": [AGENT]}

    future = _enqueue(shop, shop.context_for("p-ada"), record, posted)
    _drain(shop, future, trigger)

    assert future.exception(timeout=IMMEDIATE_S) is None
    assert shop.turn_in_flight("p-ada") is False
    assert shop.turn_state("p-ada") == IDLE_TURN
    # the dead-turn latch survives the very call that deleted the entry above
    assert shop.turn_payload("p-ada") == {
        "state": TURN_IDLE, "queuePosition": 0, "lastTurn": "failed",
    }

    logged = [r for r in caplog.records if r.name == "falkorchat.storefront"]
    assert len(logged) == 1
    assert logged[0].levelno == logging.ERROR
    assert logged[0].exc_info is not None
    assert logged[0].exc_info[0] is RuntimeError
    assert "p-ada" in logged[0].getMessage()
    assert "m-1" in logged[0].getMessage()


def test_a_storefront_with_no_trigger_still_queues_and_clears_the_turn(
    services, caplog
):
    """`trigger=None` — an app built without the workflow engine — makes the
    turn a no-op, and deliberately not a *skipped* one.

    The `409` gate, the queue accounting and both quiesce drains are properties
    of the post, not of the engine: switching them off with the engine would
    make the storefront behave differently in the one configuration nobody
    tests it in.

    **A no-op is also silent, and that is the half nothing asserted.**
    `_run_turn`'s `if self._trigger is None: return` is what makes it one:
    delete those two lines and `None.maybe_trigger` raises `AttributeError`,
    which the isolation block catches, logs and clears — satisfying every other
    assertion here, and putting an `ERROR` traceback in the log on **every**
    post in the one deployment shape nobody watches. That mutation survived at
    **280 passed** (`docs/reviews/salesperson-ui-impl.md` `## Pass 17`, P17-7,
    mutation E). The empty log below closes it; its positive control is the
    neighbouring `…_trigger_raises_is_isolated…`, which asserts the same logger
    *does* speak when a turn really dies, so "no records" cannot pass by the
    logger being misconfigured.
    """
    caplog.set_level(logging.DEBUG, logger="falkorchat.storefront")
    shop = _storefront(services)
    record = ParticipantRecord(
        participant_id="p-ada", display_name="Ada", language="en",
        channel_id="ch-ada", thread_id="th-ada", joined_at=1,
    )
    posted = {"msgId": "m-1", "threadId": "th-ada", "text": "hi",
              "role": "member", "mentions": [AGENT]}

    booking = shop.reserve_turn("p-ada")
    # the slot is occupied by the reservation, engine or no engine…
    assert shop.turn_in_flight("p-ada") is True
    future = shop.enqueue_turn(
        shop.context_for("p-ada"), record, posted, booking
    )
    future.result(timeout=IMMEDIATE_S)

    assert future.exception(timeout=IMMEDIATE_S) is None
    # …and released by the worker that did nothing with it
    assert shop.turn_in_flight("p-ada") is False
    assert [r for r in caplog.records if r.name == "falkorchat.storefront"] == []
    # a no-op turn did not die, so it must not set the latch either
    assert shop.turn_payload("p-ada")["lastTurn"] is None


class _FailOnceTrigger:
    """A `WorkflowTrigger` seam whose **first** call dies and whose second
    completes — the fixture the dead-turn latch's lifecycle tests need: one
    failed turn to set the latch, then a second, ordinary turn from the same
    participant to exercise what clears it.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.done = threading.Event()

    def maybe_trigger(self, ctx, **kwargs):  # noqa: ANN001, ANN003
        self.calls += 1
        try:
            if self.calls == 1:
                raise RuntimeError("the LLM endpoint is down")
        finally:
            self.done.set()


def test_the_dead_turn_latchs_lifecycle_set_postable_and_cleared_by_the_next_enqueue(
    services,
):
    """§5.1's S9 row's own lifecycle claim, in one test: *a participant whose
    latch is set can still post at all* (`in_flight` untouched, no `409` from
    the latch itself), *the next accepted post clears it while a `409`-refused
    post does not*.

    **Two refusal shapes are exercised on purpose, because they read the latch
    at two different moments and both must leave it standing.** A `409` is
    `reserve_turn` itself refusing — nothing about the dead turn changes when
    nothing is booked. A reservation that *is* granted but whose write then
    fails before `enqueue_turn` is ever called — `services.post_message`
    raising, in the real route — is the shape Pass 20 named explicitly: the
    booking is released, but nothing reached a worker, so the notice must
    survive that release too. Only the third post, which reaches
    `enqueue_turn` and a real `submit`, may clear it.
    """
    trigger = _FailOnceTrigger()
    shop = _storefront(services, trigger=trigger)
    record = ParticipantRecord(
        participant_id="p-ada", display_name="Ada", language="en",
        channel_id="ch-ada", thread_id="th-ada", joined_at=1,
    )
    posted = {"msgId": "m-1", "threadId": "th-ada", "text": "hi",
              "role": "member", "mentions": [AGENT]}
    ctx = shop.context_for("p-ada")

    # first turn dies, sets the latch
    first = _enqueue(shop, ctx, record, posted)
    _drain(shop, first, trigger)
    assert shop.turn_payload("p-ada")["lastTurn"] == "failed"

    # the latch does not itself gate anything — postable at once
    assert shop.turn_in_flight("p-ada") is False

    # a concurrent second tab's reservation is refused (the `409`) — untouched
    concurrent = shop.reserve_turn("p-ada")
    assert concurrent is not None
    assert shop.reserve_turn("p-ada") is None
    assert shop.turn_payload("p-ada")["lastTurn"] == "failed"

    # that reservation's write then fails and releases it, never reaching
    # `enqueue_turn` — still untouched, which is Pass 20's own named case
    assert shop.release_turn("p-ada", concurrent) is True
    assert shop.turn_payload("p-ada")["lastTurn"] == "failed"

    # the next post is accepted and reaches the worker — *now* it clears
    second = _enqueue(shop, ctx, record, {**posted, "msgId": "m-2"})
    _drain(shop, second, trigger)
    assert second.exception(timeout=IMMEDIATE_S) is None
    assert shop.turn_payload("p-ada")["lastTurn"] is None


def test_a_write_failure_after_a_grant_reservation_leaves_the_latch_standing(
    services,
):
    """The narrower, single-purpose form of the case above — reserve
    succeeds, the write never happens, `enqueue_turn` is never called — kept
    as its own test because it is the exact shape
    `docs/plans/salesperson-ui.md` §5.1's S9 row names for the ordering
    mutant: clearing in `reserve_turn` instead of `enqueue_turn` passes the
    lifecycle test above right up to this one line, because that test's
    `reserve_turn` calls happen on a *different* participant's booking. Here
    the same participant whose latch is already set reserves again — under
    the wrong placement that reservation alone would wipe the notice, before
    anything is known about whether the write it is for will even succeed.
    """
    trigger = _RecordingTrigger(explode=True)
    shop = _storefront(services, trigger=trigger)
    record = ParticipantRecord(
        participant_id="p-ada", display_name="Ada", language="en",
        channel_id="ch-ada", thread_id="th-ada", joined_at=1,
    )
    posted = {"msgId": "m-1", "threadId": "th-ada", "text": "hi",
              "role": "member", "mentions": [AGENT]}
    ctx = shop.context_for("p-ada")

    _drain(shop, _enqueue(shop, ctx, record, posted), trigger)
    assert shop.turn_payload("p-ada")["lastTurn"] == "failed"

    # reserve again for the same participant — as far as the write knows this
    # is about to fail (`services.post_message` raising in the real route)
    booking = shop.reserve_turn("p-ada")
    assert booking is not None
    # …and the write fails, releasing it without ever calling `enqueue_turn`
    assert shop.release_turn("p-ada", booking) is True

    assert shop.turn_payload("p-ada")["lastTurn"] == "failed"


def _refuses_to_start(self):  # noqa: ANN001, ANN201, ARG001
    """Stands in for `threading.Thread.start` under thread exhaustion.

    CPython raises exactly this from `t.start()` inside
    `ThreadPoolExecutor._adjust_thread_count`
    (`/usr/lib/python3.12/concurrent/futures/thread.py:202`) when the OS
    refuses a thread — the one shape that cannot be reproduced honestly by
    exhausting the machine inside a unit test.
    """
    raise RuntimeError("can't start new thread")


def _blocking_trigger():
    """A trigger that parks its worker inside `maybe_trigger` until released.

    `(trigger, entered, gate)`. A gate rather than a sleep for the reason the
    rest of this section gives: a sleep asserts in-flight-ness by hoping, a
    gate asserts it.
    """
    entered = threading.Event()
    gate = threading.Event()

    class _Blocking:
        def maybe_trigger(self, ctx, **kwargs):  # noqa: ANN001, ANN003, ARG002
            entered.set()
            gate.wait(timeout=IMMEDIATE_S)

    return _Blocking(), entered, gate


def _queue_holding_trigger():
    """`_blocking_trigger`'s sibling for the S9b tests: it parks the **first**
    turn on the worker, lets every later one run, and **records which turns
    reached it at all**.

    `(trigger, entered, gate, later, later_gate)`. With `turn_workers=1` the
    parked turn is what makes a second post's work item provably `PENDING` in
    the executor's queue rather than merely "probably not started yet" — the
    only state `Future.cancel()` can act on. `trigger.ran` is the list of
    `msg_id`s that reached the workflow layer, and it is the assertion that
    separates a turn that was *cancelled* from one that ran and cleared up
    quietly: a map entry can be gone either way.

    `later` fires when a turn **after** the first reaches a worker, which is how
    a test knows a future has left `PENDING`. `later_gate` starts **set**, so by
    default those turns run straight through; a test that needs one held
    *inside* the trigger clears it first and sets it when done.
    """
    entered = threading.Event()
    gate = threading.Event()
    later = threading.Event()
    later_gate = threading.Event()
    later_gate.set()

    class _Holding:
        def __init__(self) -> None:
            self.ran: list[str] = []

        def maybe_trigger(self, ctx, **kwargs):  # noqa: ANN001, ANN003, ARG002
            self.ran.append(kwargs["msg_id"])
            if entered.is_set():
                later.set()
                later_gate.wait(timeout=IMMEDIATE_S)
                return
            entered.set()
            gate.wait(timeout=IMMEDIATE_S)

    return _Holding(), entered, gate, later, later_gate


def _posted(msg_id: str, thread_id: str):
    """The shape `enqueue_turn` carries onto the worker, for the tests whose
    subject is the queue rather than the graph write that fills it in."""
    return {"msgId": msg_id, "threadId": thread_id, "text": "hi",
            "role": "member", "mentions": [AGENT]}


def _participant(pid: str) -> ParticipantRecord:
    return ParticipantRecord(
        participant_id=pid, display_name=pid, language="en",
        channel_id=f"ch-{pid}", thread_id=f"th-{pid}", joined_at=1,
    )


def test_a_running_turn_holds_a_worker_and_the_one_behind_it_is_first_in_line(
    services,
):
    """The two readings that separate a running turn from a waiting one, taken
    at the one instant where they differ.

    The booking is no longer taken here at all — `reserve_turn` writes the
    entry on the request thread, *before* the message write, so "book before
    submit" is now structural rather than a race S9a had to win. What is left
    to measure is the worker's own write and the number derived from it: the
    running participant reads `thinking`/`0`, and the one behind them reads
    `queued`/**`0`** — *first in line*, because a running turn occupies a
    worker rather than a place in the line (§5.2). The old spelling of this
    test asserted `queued`/`1` for exactly this arrangement, which the current
    definition falsifies.

    A `_run_turn` that flipped the wrong entry, or one whose flip was
    unconditional and landed on a booking it no longer owned, fails the first
    of the two — and the second moves with it, since a `thinking` turn is not
    counted and a `queued` one is.
    """
    trigger, entered, gate = _blocking_trigger()
    shop = _storefront(services, trigger=trigger, turn_workers=1)
    posted = {"msgId": "m-1", "threadId": "th-x", "text": "hi",
              "role": "member", "mentions": [AGENT]}

    def record(pid):
        return ParticipantRecord(
            participant_id=pid, display_name=pid, language="en",
            channel_id=f"ch-{pid}", thread_id=f"th-{pid}", joined_at=1,
        )

    first_booking = shop.reserve_turn("p-a")
    assert shop.turn_payload("p-a") == {
        "state": TURN_QUEUED, "queuePosition": 0,
        "lastTurn": None,
    }
    first = shop.enqueue_turn(
        shop.context_for("p-a"), record("p-a"), posted, first_booking
    )
    assert entered.wait(timeout=IMMEDIATE_S), "the first turn never reached a worker"

    second_booking = shop.reserve_turn("p-b")
    second = shop.enqueue_turn(
        shop.context_for("p-b"), record("p-b"), posted, second_booking
    )

    assert second_booking.ordinal > first_booking.ordinal
    assert shop.turn_payload("p-a") == {
        "state": TURN_THINKING, "queuePosition": 0,
        "lastTurn": None,
    }
    assert shop.turn_payload("p-b") == {
        "state": TURN_QUEUED, "queuePosition": 0,
        "lastTurn": None,
    }

    gate.set()
    first.result(timeout=IMMEDIATE_S)
    second.result(timeout=IMMEDIATE_S)
    assert shop.turn_in_flight("p-a") is False
    assert shop.turn_in_flight("p-b") is False


def test_a_live_workers_finally_does_not_clear_the_booking_that_replaced_its_own(
    services,
):
    """P17-1's ownership half, with a **real worker** rather than a map probe.

    `presenter_reset_all`'s `clear_all_turns()` empties the map while workers
    are still running (`docs/reviews/salesperson-ui-impl.md` `## Pass 17`,
    Ruling 2), and the participant can then post again and be accepted, because
    the map is empty. Two bookings for one participant now exist, and the map
    holds one slot: an unconditional `finally` deletes the **second**, live
    one, at which point `turn_in_flight` is `False` and `_await_quiesce`
    returns `True` under a running turn — the state the quiesce order exists to
    make impossible, reproduced in Pass 17 exactly this way.

    **The same case pins the ordinal**, which is why both assertions are here:
    the fresh booking's ordinal must be strictly greater than the wiped one's,
    and a counter derived from the map's size restarts at `0` after the wipe —
    so it fails `>` *and* makes the old worker's ownership check pass against
    the new booking, deleting it. One mutation, two red assertions.
    """
    trigger, entered, gate = _blocking_trigger()
    shop = _storefront(services, trigger=trigger, turn_workers=1)
    record = ParticipantRecord(
        participant_id="p-ada", display_name="Ada", language="en",
        channel_id="ch-ada", thread_id="th-ada", joined_at=1,
    )
    posted = {"msgId": "m-1", "threadId": "th-ada", "text": "hi",
              "role": "member", "mentions": [AGENT]}

    old = shop.reserve_turn("p-ada")
    future = shop.enqueue_turn(shop.context_for("p-ada"), record, posted, old)
    assert entered.wait(timeout=IMMEDIATE_S), "the turn never reached a worker"

    shop.clear_all_turns()
    fresh = shop.reserve_turn("p-ada")
    assert fresh is not None, "the wipe must leave the participant postable"
    assert fresh.ordinal > old.ordinal

    gate.set()
    future.result(timeout=IMMEDIATE_S)

    assert shop.turn_state("p-ada").booking == fresh
    assert shop.turn_in_flight("p-ada") is True


def test_a_submit_refused_after_shutdown_releases_the_reservation(services):
    """P17-3, reproduced and closed.

    `shutdown_turns()` makes `submit` raise `RuntimeError: cannot schedule new
    futures after shutdown`, and S9a booked before submitting with no guard, so
    the entry survived as a turn no worker would ever clear: that participant
    `409`-refused for the life of the process, and every reset-mine of theirs
    answering `503 quiesce_timeout`. The reachable trigger is narrow; the blast
    radius is not.

    The `RuntimeError` still propagates — the reservation is released, not the
    failure — so the route's own `except` is what turns it into a response.
    """
    shop = _storefront(services)
    shop.shutdown_turns()
    record = ParticipantRecord(
        participant_id="p-ada", display_name="Ada", language="en",
        channel_id="ch-ada", thread_id="th-ada", joined_at=1,
    )
    posted = {"msgId": "m-1", "threadId": "th-ada", "text": "hi",
              "role": "member", "mentions": [AGENT]}

    booking = shop.reserve_turn("p-ada")
    assert shop.turn_in_flight("p-ada") is True

    with pytest.raises(RuntimeError):
        shop.enqueue_turn(shop.context_for("p-ada"), record, posted, booking)

    assert shop.turn_state("p-ada") == IDLE_TURN
    assert shop.turn_in_flight("p-ada") is False
    # and the participant is postable again rather than locked out
    assert shop.reserve_turn("p-ada") is not None


def test_a_submit_that_raises_after_it_queued_the_item_leaves_the_booking_standing(
    services, caplog
):
    """P20-1, and the half a shutdown-shaped test cannot reach.

    `ThreadPoolExecutor.submit` puts the work item on the shared queue
    (`/usr/lib/python3.12/concurrent/futures/thread.py:178`) **before** it
    calls `_adjust_thread_count` (`:179`), whose `t.start()` (`:202`) is the
    only place `RuntimeError("can't start new thread")` comes from. So a
    thread-exhaustion refusal raises with the turn already queued, and the
    worker that is busy right now will run it as soon as it is free —
    `_worker`'s loop (`:69-95`) never checks who put an item on that queue.
    Releasing the booking on that shape manufactures a live turn the map says
    is not there: `turn_in_flight` `False` under a running turn, which is
    §4.4 measure 1a's invariant and the state `_await_quiesce` exists to make
    impossible (`docs/reviews/salesperson-ui-impl.md` `## Pass 20`, P20-1,
    reproduced end to end against the real `Storefront`).

    **Both halves are asserted, and neither alone is the test.** The booking
    must still be standing at the instant `enqueue_turn` raises — that is what
    goes red on an implementation that releases from a bare `except` around
    `submit` — *and* the queued item must then run and clear its own booking in
    `_run_turn`'s `finally`, which is what makes the standing booking a
    correct hand-off rather than the leak P17-3 describes. A test that read
    only the end state would pass against the defect: both implementations
    end idle.

    Mutation-tested: reverting `enqueue_turn` to release inside the `except`
    reddens this test on the `turn_in_flight(...) is True` line, and leaves
    `test_a_submit_refused_after_shutdown_releases_the_reservation` green —
    which is the whole reason this case exists beside it.
    """
    caplog.set_level(logging.DEBUG, logger="falkorchat.storefront")

    class _BlockingRecorder:
        """Parks the first turn until released; records every turn it ran."""

        def __init__(self):
            self.entered = threading.Event()
            self.gate = threading.Event()
            self._lock = threading.Lock()
            self.seen: list[str] = []

        def maybe_trigger(self, ctx, **kwargs):  # noqa: ANN001, ANN003, ARG002
            with self._lock:
                self.seen.append(kwargs["msg_id"])
            self.entered.set()
            self.gate.wait(timeout=IMMEDIATE_S)

    trigger = _BlockingRecorder()
    # Two workers, one of them busy: `_adjust_thread_count` then has no idle
    # thread to reuse and room to make a new one, so it reaches `t.start()` —
    # which is the only line the patch below can intercept.
    shop = _storefront(services, trigger=trigger, turn_workers=2)

    def record(pid):
        return ParticipantRecord(
            participant_id=pid, display_name=pid, language="en",
            channel_id=f"ch-{pid}", thread_id=f"th-{pid}", joined_at=1,
        )

    def posted(pid, msg_id):
        return {"msgId": msg_id, "threadId": f"th-{pid}", "text": "hi",
                "role": "member", "mentions": [AGENT]}

    busy = shop.reserve_turn("p-a")
    first = shop.enqueue_turn(
        shop.context_for("p-a"), record("p-a"), posted("p-a", "m-a"), busy
    )
    assert trigger.entered.wait(timeout=IMMEDIATE_S), "no worker is busy"

    booking = shop.reserve_turn("p-b")
    assert booking is not None
    original_start = threading.Thread.start
    try:
        threading.Thread.start = _refuses_to_start  # type: ignore[method-assign]
        with pytest.raises(RuntimeError):
            shop.enqueue_turn(
                shop.context_for("p-b"), record("p-b"), posted("p-b", "m-b"),
                booking,
            )
    finally:
        threading.Thread.start = original_start  # type: ignore[method-assign]

    # Half one: the booking is standing at the instant the refusal propagates.
    assert shop.turn_in_flight("p-b") is True
    assert shop.turn_state("p-b").booking == booking
    # ...and the refusal is on the record, with the two identifiers an operator
    # needs to tie it to a turn — this log is the only trace it leaves.
    refusals = [
        r for r in caplog.records
        if r.name == "falkorchat.storefront" and "submit refused" in r.getMessage()
    ]
    assert len(refusals) == 1
    assert refusals[0].levelno == logging.ERROR
    assert "p-b" in refusals[0].getMessage()
    assert f"ordinal={booking.ordinal}" in refusals[0].getMessage()

    # Half two: the item that was queued anyway runs on the freed worker and
    # clears its own booking, so the standing entry was a hand-off, not a leak.
    trigger.gate.set()
    first.result(timeout=IMMEDIATE_S)
    deadline = time.monotonic() + IMMEDIATE_S
    while shop.turn_in_flight("p-b") and time.monotonic() < deadline:
        time.sleep(0.005)
    assert shop.turn_in_flight("p-b") is False
    assert shop.turn_state("p-b") == IDLE_TURN
    # the turn really ran; it was not cleared by some other path
    assert trigger.seen == ["m-a", "m-b"]


def test_the_flag_alone_refuses_the_turn_while_the_executor_is_still_alive(
    services,
):
    """**The case that separates the delivered placement from the one v1.30
    rejected**, and the only one that does.

    v1.30 spends a paragraph on why reading `_turns_shutdown` *inside* the
    `except` around `submit` is strictly weaker than reading it before the
    call, and `enqueue_turn`'s docstring repeats the argument — but neither of
    the two cases beside this one can tell the placements apart.
    `test_a_submit_refused_after_shutdown_releases_the_reservation` calls the
    real `shutdown_turns()`, so `submit` refuses under either placement; the
    exhaustion case runs with the flag `False`, so an `except` that consults it
    releases nothing either way. Measured: with the flag read moved into the
    `except`, both of them stayed green and the suite's only reaction was the
    raises guard in `tests/test_storefront_api.py` — reddening because the
    mutant deletes the literal `raise`, with a failure message that invites
    dropping `"RuntimeError"` from an allowlist
    (`docs/reviews/salesperson-ui-impl.md` `## Pass 21`, P21-3, Appendix Q §4).

    **The window is the one `shutdown_turns()` opens between its own two
    statements**: the flag is set and `executor.shutdown(wait=True)` has not
    run yet, so `submit` would succeed. That is exactly where the two
    placements differ — before the call, the post is refused having queued
    nothing; inside the `except`, there is no exception to consult the flag
    from, so the turn is submitted and the refusal never happens.

    Setting `_turns_shutdown` directly is the point rather than a shortcut:
    calling `shutdown_turns()` would close the executor too and collapse the
    window back onto the case above. The `_executor._shutdown` assertion above
    is this test's **positive control** — it says the refusal came from the
    flag and not from a stopped pool, which is what makes the case
    discriminating rather than a second spelling of the shutdown test.

    **`_work_queue.qsize() == 0` does not prove nothing was submitted** — it
    proves nothing was *left over*, and those are different on a live
    executor. `submit` starts a worker before this assertion runs
    (`_adjust_thread_count`, CPython 3.12.3
    `/usr/lib/python3.12/concurrent/futures/thread.py:179`), and that worker
    drains the queue, so `qsize()` reads `0` on both sides of the refusal —
    measured: with the flag read moved to *after* `submit`, `qsize()` is still
    `0` even though the item **was** queued and run
    (`docs/reviews/salesperson-ui-impl.md` `## Pass 22`, P22-1). `len(
    shop._executor._threads)` is the actual submit-detector: a
    `ThreadPoolExecutor` starts its first worker inside `submit` and
    `_threads` never shrinks before `shutdown`, so it reads `0` iff no submit
    ever happened. That equivalence holds **only for an executor nothing has
    ever been submitted to** — this fixture's, always, since it is built fresh
    per test and this case is the first and only call into `enqueue_turn` on
    it. A fixture that submitted first would make this assertion fail loudly
    rather than pass by accident, which is why that is a caveat to record
    here rather than a reason to pick a different assertion.
    """
    shop = _storefront(services)
    record = ParticipantRecord(
        participant_id="p-ada", display_name="Ada", language="en",
        channel_id="ch-ada", thread_id="th-ada", joined_at=1,
    )
    posted = {"msgId": "m-1", "threadId": "th-ada", "text": "hi",
              "role": "member", "mentions": [AGENT]}

    booking = shop.reserve_turn("p-ada")
    shop._turns_shutdown = True  # noqa: SLF001 — the window, without closing the pool

    # the positive control: the pool would have accepted this turn
    assert shop._executor._shutdown is False, (  # noqa: SLF001
        "the window this case pins needs a live executor; a stopped one makes "
        "it a duplicate of the post-shutdown case"
    )

    with pytest.raises(RuntimeError):
        shop.enqueue_turn(shop.context_for("p-ada"), record, posted, booking)

    # ...nothing was left over (does not by itself prove nothing was
    # submitted — see the docstring)...
    assert shop._executor._work_queue.qsize() == 0  # noqa: SLF001
    # ...and nothing was submitted: no worker was ever started
    assert len(shop._executor._threads) == 0  # noqa: SLF001
    # ...and the reservation went with the refusal, since nothing will clear it
    assert shop.turn_in_flight("p-ada") is False
    assert shop.turn_state("p-ada") == IDLE_TURN
    assert shop.reserve_turn("p-ada") is not None


def test_shutdown_turns_is_idempotent(services):
    """The lifespan calls it once; a second call must not raise, so a test (or
    a double shutdown) cannot turn an orderly stop into a traceback."""
    shop = _storefront(services)
    shop.shutdown_turns()
    shop.shutdown_turns()


# ── configuration (§4.9's one-workspace-variable rule) ───────────────────────


_CONFIG_SOURCE = Path(config.__file__).read_text(encoding="utf-8")
_REPO_ROOT = Path(__file__).resolve().parents[2]  # falkor-chat/
_PACKAGE_DIR = Path(__file__).resolve().parents[1] / "falkorchat"


def test_config_reads_exactly_the_documented_storefront_env_vars():
    """The seven S6 names, spelled once each, in the one module that reads them.

    Six take the `FALKORCHAT_STOREFRONT_` prefix and `FALKORCHAT_THREAD_LIMIT`
    does not — the presenter key included, so the delivered spelling is
    `FALKORCHAT_STOREFRONT_PRESENTER_KEY`. The justification is deliberately
    **in-repo and checkable**: this set is exactly what `config.py` reads and
    exactly what `docs/SERVER.md` §1.3's table documents, so a rename that
    updates one and not the other reddens here. It is *not* justified by
    quoting the plan — a docstring that cites another document as its authority
    inherits that document's drift, which is the failure this coordination has
    now produced three times.
    """
    expected = {
        "FALKORCHAT_STOREFRONT_ENABLED",
        "FALKORCHAT_STOREFRONT_DIR",
        "FALKORCHAT_STOREFRONT_PRESENTER_KEY",
        "FALKORCHAT_STOREFRONT_TURN_WORKERS",
        "FALKORCHAT_STOREFRONT_QUIESCE_S",
        "FALKORCHAT_STOREFRONT_LOCALES",
    }
    read = set(re.findall(r'"(FALKORCHAT_[A-Z_]+)"', _CONFIG_SOURCE))

    assert {name for name in read if "STOREFRONT" in name} == expected
    assert "FALKORCHAT_THREAD_LIMIT" in read

    # The doc half, asserted rather than asserted-in-prose: every name the code
    # reads is documented, so renaming one in `config.py` without sweeping
    # `SERVER.md` reddens here instead of drifting silently. Precedent for a
    # test reaching outside the package: `test_seed_workflows_script.py`, which
    # pins a `scripts/` invariant the same way.
    server_md = (_REPO_ROOT / "docs" / "SERVER.md").read_text(encoding="utf-8")
    undocumented = sorted(
        name for name in expected | {"FALKORCHAT_THREAD_LIMIT"}
        if name not in server_md
    )
    assert undocumented == []


def _modules_mentioning(needle: str) -> list[str]:
    """Every module in the package whose source contains `needle`.

    Shared by the tripwire below and its control so the two run the *identical*
    scan — a control that walks the tree by some other route would not prove the
    tripwire's own walk found anything.
    """
    return sorted(
        str(path.relative_to(_PACKAGE_DIR))
        for path in _PACKAGE_DIR.rglob("*.py")
        if needle in path.read_text(encoding="utf-8")
    )


def test_no_second_workspace_variable_exists_anywhere_in_the_package():
    """§4.9 move 2: the storefront's workspace **is** `config.WS_ID`. B3 was only
    possible because two variables could disagree; with one, the
    misconfiguration is not expressible. This is the tripwire against
    reintroducing `FALKORCHAT_DEMO_WS` by reflex.

    **The control is the first assertion, and it is not decoration.** As shipped
    this test asserted only emptiness, and `Path.rglob` on a *missing* directory
    yields nothing and raises nothing — so it passed identically whether it
    scanned 27 modules or zero (`docs/reviews/salesperson-ui-impl.md` Pass 6,
    S6-3: `_PACKAGE_DIR` repointed at a nonexistent path, still green). Pinning a
    string that **must** be found, through the same scan, is what makes the
    emptiness below a finding rather than an absence of evidence.
    """
    assert _modules_mentioning("FALKORCHAT_STOREFRONT_ENABLED") == ["config.py"]

    assert _modules_mentioning("FALKORCHAT_DEMO_WS") == []


def test_dev_surface_has_no_environment_variable():
    """§4.9 move 1: `dev_surface` is a `create_app` parameter and nothing else,
    so no operator setting can put the legacy unauthenticated surface back while
    participants exist."""
    assert "DEV_SURFACE" not in _CONFIG_SOURCE
    assert not re.search(r'environ\.get\(\s*"[^"]*DEV_SURFACE', _CONFIG_SOURCE)


def test_env_csv_falls_back_rather_than_yielding_an_empty_locale_set(monkeypatch):
    """An empty locale tuple would reject every language a participant could
    pick — a typo in the operator's shell must not become a demo nobody can join
    (`config.STOREFRONT_LOCALES` is this helper's only consumer)."""
    default = ("en", "pt-BR", "es")

    monkeypatch.delenv("FALKORCHAT_STOREFRONT_LOCALES", raising=False)
    assert config._env_csv("FALKORCHAT_STOREFRONT_LOCALES", default) == default

    for blank in ("", "   ", ",", " , , "):
        monkeypatch.setenv("FALKORCHAT_STOREFRONT_LOCALES", blank)
        assert config._env_csv("FALKORCHAT_STOREFRONT_LOCALES", default) == default

    monkeypatch.setenv("FALKORCHAT_STOREFRONT_LOCALES", " en , de,, fr ")
    assert config._env_csv("FALKORCHAT_STOREFRONT_LOCALES", default) == ("en", "de", "fr")


def test_an_unset_presenter_key_is_reported_as_unconfigured(services):
    """`hmac.compare_digest("", "")` is `True`, so S10's login must reject an
    unset key *before* comparing — otherwise an unconfigured deployment hands
    the reset-everyone button to whoever posts an empty key first."""
    assert _storefront(services, presenter_key="").presenter_configured is False
    assert _storefront(services, presenter_key="s3cret").presenter_configured is True


def test_the_storefront_context_is_the_participants_own_scope(seeded):
    """§4.3: `ctx.actor == participantId == customerId`, and `ws` is the one
    workspace variable — no route accepts either from a client."""
    ctx = seeded.context_for("p-abc")

    assert ctx.ws == WS
    assert ctx.actor == "p-abc"


def test_the_workspace_defaults_to_the_single_config_variable(services):
    """Constructed without an explicit `ws`, the storefront uses `config.WS_ID`
    — there is no second workspace value for it to disagree with (§4.9)."""
    shop = Storefront(
        services, presenter_key="k", turn_workers=1, quiesce_s=1.0,
    )

    assert shop.ws == config.WS_ID
    assert shop.locales == config.STOREFRONT_LOCALES


# ═══════════════════════════════════════════════════════════════════════════
# S7 — state, reset, catalog, images, order lifecycle
# ═══════════════════════════════════════════════════════════════════════════

# The catalog lives in the **global** `reference` graph (`docs/QUERIES.md` §15),
# which has no repository write method — it is seed-script-only
# (`scripts/seed_catalog.sh`). Fixtures are therefore a raw, test-only write,
# the same posture `tests/test_repository.py::_seed_products` takes, and every
# catalog test goes through `catalog_repo` so `reference`'s node data is wiped
# (its schema — the `Product` index/constraint pair — survives a DETACH DELETE).


def _catalog_rows(n: int) -> list[dict]:
    """`n` synthetic products with `seed_catalog.sh`-shaped deterministic slugs,
    priced so `price ASC` is `p-001 … p-0nn`."""
    return [
        {
            "productId": f"widget-{i:03d}", "name": f"Widget {i:03d}",
            "nameNormalized": f"widget {i:03d}", "category": "Accessories",
            "categoryNormalized": "accessories", "price": float(10 + i),
        }
        for i in range(1, n + 1)
    ]


def _seed_catalog(conn, rows):
    db.reference_graph(conn).query(
        "UNWIND $rows AS row "
        "CREATE (:Product {productId: row.productId, name: row.name, "
        "                  nameNormalized: row.nameNormalized, "
        "                  category: row.category, "
        "                  categoryNormalized: row.categoryNormalized, "
        "                  price: row.price})",
        {"rows": rows},
    )
    return rows


def _ticking_clock(start: int = 1_700_000_000_000):
    """A strictly increasing ms clock for `Services`.

    `Order.placedAt` ties break by `orderId DESC`, and `orderId` is a `uuid4`
    — so two orders placed inside the same millisecond would make "the most
    recently placed order" a coin flip. Every timestamp distinct removes the
    tie rather than betting on the wall clock ticking between two calls.
    """
    counter = iter(range(start, start + 1_000_000))
    return lambda: next(counter)


@pytest.fixture()
def catalog_repo(conn, wf_repo):
    """`wf_repo`, plus a teardown that leaves `reference` **empty**.

    `wf_repo` wipes `reference` on *setup* only, so whichever test touches it
    last leaves its fixture products behind in a **global** graph — and
    `scripts/seed_catalog.sh` `MERGE`s by `productId`, so a stray `widget-…`
    survives the re-seed a default `pytest` run already obliges and then makes
    `scripts/verify_catalog.sh` report a catalog mismatch (17 products,
    expected 15) to whoever runs it next (`falkor-chat/AGENTS.md`).
    """
    yield wf_repo
    db.reference_graph(conn).query("MATCH (n) DETACH DELETE n")


@pytest.fixture()
def stocked(conn, catalog_repo):
    """A `Storefront` on a wiped `ws:test` **and** a wiped `reference`, with the
    demo `Agent` registered. Catalog seeding is per-test."""
    catalog_repo.ensure_agent(WS, agent_id=AGENT, name="Demo agent", created_at=90)
    return _storefront(Services(catalog_repo, clock=_ticking_clock()))


def _assets(root: Path, names) -> Path:
    """A fixture asset directory in the served shape: `<root>/products/<file>`."""
    products = root / "products"
    products.mkdir(parents=True, exist_ok=True)
    for name in names:
        (products / name).write_bytes(b"\x00")
    return root


# ── get_state (§5.2) ─────────────────────────────────────────────────────────


def test_get_state_reports_profile_cart_order_and_turn(stocked, conn):
    _seed_catalog(conn, _catalog_rows(3))
    record = stocked.join("Ada", "pt-BR")
    ctx = stocked.context_for(record.participant_id)
    services = stocked._services
    services.save_profile(ctx, delivery_address="12 Rua das Flores")
    services.add_cart_item(ctx, product_name="Widget 001", quantity=2)
    services.add_cart_item(ctx, product_name="Widget 002", quantity=1)
    # Two *other* participants booked ahead of Ada, so her derived position is
    # `2` — the number cannot be fabricated any more, it has to be earned.
    stocked.reserve_turn("p-ahead-1")
    stocked.reserve_turn("p-ahead-2")
    stocked.reserve_turn(record.participant_id)

    state = stocked.get_state(ctx)

    assert set(state) == {"profile", "cart", "order", "turn"}
    assert state["profile"] == {
        "name": "Ada", "deliveryAddress": "12 Rua das Flores",
    }
    assert [line["name"] for line in state["cart"]["items"]] == [
        "Widget 001", "Widget 002",
    ]
    assert state["cart"]["total"] == pytest.approx(11.0 * 2 + 12.0)
    assert state["order"] is None
    assert state["turn"] == {"state": TURN_QUEUED, "queuePosition": 2, "lastTurn": None}


def test_get_state_of_a_fresh_participant_is_the_join_shape(stocked, conn):
    """The join name is already in the profile (§4.10) — everything else empty.

    The positive control for the reset-parity test below: `name` is `"Ada"`
    here *and* after a self-reset, and `None` only if the profile write is
    missing.
    """
    _seed_catalog(conn, _catalog_rows(1))
    record = stocked.join("Ada", "en")

    state = stocked.get_state(stocked.context_for(record.participant_id))

    assert state == {
        "profile": {"name": "Ada", "deliveryAddress": None},
        "cart": {"items": [], "total": 0},
        "order": None,
        "turn": {"state": TURN_IDLE, "queuePosition": 0, "lastTurn": None},
    }


def test_get_states_order_block_is_the_repository_read_not_a_local_composition(
    stocked, conn
):
    """§5.1's S7 row: the order block comes from `services.get_current_order`.

    Written so a storefront-side reconstruction cannot pass. Two orders are
    placed, the cart is refilled after each, and the frozen line of the
    **older** order names a product the current cart no longer holds — so
    anything composed here from cart/profile parts reports the wrong order, the
    wrong lines, or both. The repository read answers "most recently *placed*,
    whatever its status" (`docs/QUERIES.md` §18.8), which is the second order
    with its own frozen line, while the live cart holds a third product.
    """
    _seed_catalog(conn, _catalog_rows(3))
    record = stocked.join("Ada", "en")
    ctx = stocked.context_for(record.participant_id)
    services = stocked._services
    services.add_cart_item(ctx, product_name="Widget 001", quantity=1)
    first = services.place_order(ctx)
    services.add_cart_item(ctx, product_name="Widget 002", quantity=3)
    second = services.place_order(ctx)
    services.add_cart_item(ctx, product_name="Widget 003", quantity=1)

    order = stocked.get_state(ctx)["order"]

    assert order is not None
    assert order["orderId"] == second["orderId"]
    assert order["orderId"] != first["orderId"]
    assert order["status"] == "placed"
    assert [(line["productId"], line["quantity"]) for line in order["lines"]] == [
        ("widget-002", 3)
    ]
    assert order["total"] == pytest.approx(12.0 * 3)
    # …and it is exactly what the repository read returns, field for field.
    assert order == services.get_current_order(ctx)


def test_get_state_is_scoped_to_the_calling_participant(stocked, conn):
    _seed_catalog(conn, _catalog_rows(2))
    ada = stocked.join("Ada", "en")
    bob = stocked.join("Bob", "es")
    ada_ctx = stocked.context_for(ada.participant_id)
    bob_ctx = stocked.context_for(bob.participant_id)
    stocked._services.add_cart_item(ada_ctx, product_name="Widget 001", quantity=4)
    stocked._services.place_order(ada_ctx)
    _pin_thinking(stocked, ada.participant_id)

    bob_state = stocked.get_state(bob_ctx)

    assert bob_state["profile"] == {"name": "Bob", "deliveryAddress": None}
    assert bob_state["cart"] == {"items": [], "total": 0}
    assert bob_state["order"] is None
    assert bob_state["turn"] == {"state": TURN_IDLE, "queuePosition": 0, "lastTurn": None}
    # …while Ada's own state is unaffected by having been read past.
    assert stocked.get_state(ada_ctx)["order"]["total"] == pytest.approx(44.0)


# ── list_catalog + the image manifest (§4.7) ─────────────────────────────────


def test_list_catalog_returns_all_fifteen_rows(stocked, conn):
    _seed_catalog(conn, _catalog_rows(15))

    rows = stocked.list_catalog()

    assert len(rows) == 15
    assert [row["productId"] for row in rows] == [
        f"widget-{i:03d}" for i in range(1, 16)
    ]
    assert all(
        set(row) == {"productId", "name", "category", "price", "imageUrl"}
        for row in rows
    )
    assert rows[0] == {
        "productId": "widget-001", "name": "Widget 001",
        "category": "Accessories", "price": 11.0, "imageUrl": None,
    }


class _CountingReferenceGraph:
    """A `Graph` proxy that records every Cypher string sent through it."""

    def __init__(self, graph, log: list[str]) -> None:
        self._graph = graph
        self._log = log

    def query(self, cypher, *args, **kwargs):
        self._log.append(cypher)
        return self._graph.query(cypher, *args, **kwargs)

    def ro_query(self, cypher, *args, **kwargs):
        self._log.append(cypher)
        return self._graph.ro_query(cypher, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._graph, name)


def _read_summary(log: list[str]) -> str:
    """The read log as one entry per distinct query with its repeat count —
    a `1 + n` shows up as `1x MATCH (p:Product) WHERE …; 15x MATCH (p:Product
    {nameNormalized: …` rather than fifteen near-identical 200-char lines."""
    counts: dict[str, int] = {}
    for cypher in log:
        key = " ".join(cypher.split())[:70]
        counts[key] = counts.get(key, 0) + 1
    return "; ".join(f"{n}x {query}…" for query, n in counts.items())


@pytest.fixture()
def reference_reads(monkeypatch) -> list[str]:
    """Every Cypher string sent to the global `reference` graph, in order.

    Patched at **`db.reference_graph`** — the single seam
    `Repository._reference` resolves through (`repository.py:173`) — rather
    than at a method or attribute on a particular object. So the count is of
    real graph round trips, whoever issued them: `self._services`,
    `self._repo`, or a `Repository` constructed on the spot. That is the whole
    point of counting rather than patching a name; see
    `test_the_catalog_is_read_once_not_once_per_product`.
    """
    log: list[str] = []
    real = db.reference_graph
    monkeypatch.setattr(
        db, "reference_graph", lambda conn: _CountingReferenceGraph(real(conn), log)
    )
    return log


def test_the_catalog_is_read_once_not_once_per_product(
    conn, catalog_repo, reference_reads
):
    """S7c's binding test: the `productId` projection and the removal of S7's
    `1 + n` workaround cannot ship apart (`docs/plans/salesperson-ui.md` §5.1
    S7c).

    S7 listed the catalog with `services.filter_products` — which projected no
    `productId` — and recovered each row's id with a second `lookup_product`
    point read, dropping any row that failed to re-resolve. S7c widened the
    projection instead. Three assertions bind that, and none of them names a
    method:

    * **The rows.** Revert the projection and `list_catalog` raises
      `KeyError: 'productId'` at `manifest.get(row["productId"])`. One of the
      15 products (`opaque-sku-42`) carries a `productId` that is **not** the
      slug of its name, so a `_catalog_rows` that re-derived the id from
      `row["name"]` — the name-slugify alternative S7's docstring weighed and
      rejected — fails here too, even though `seed_catalog.sh` does slugify
      names today and every other fixture row would let it pass.
    * **The read count is real.** `1 <= reads` fails if the spy stops
      observing anything, which is what makes the equality below meaningful
      rather than vacuously `0 == 0`.
    * **The read count does not grow with the catalog.** Listing 15 products
      costs exactly what listing 3 costs. *This* is "read once, not once per
      product", and unlike patching `services.lookup_product` to raise it does
      not care which attribute the second read goes through — a `1 + n` routed
      through `self._repo`, or through any method at all, moves the count.

    **Why the bound is 2 and not 1:** `list_catalog` calls `_catalog_rows`
    twice on a cold instance (once via `build_image_manifest`, once for the
    rows), so the constant is 2 — known, out of S7c's scope, and S9's to
    collapse when it takes `storefront.py` next. The equality is the load-
    bearing assertion; the bound only pins the constant small and would
    survive S9 tightening it to 1.
    """
    catalog_repo.ensure_agent(WS, agent_id=AGENT, name="Demo agent", created_at=90)

    def _cold_list_catalog(size: int):
        """Seed a `size`-product catalog and list it on a fresh `Storefront`,
        returning the seeded rows, the listing, and the `reference` read count.
        """
        db.reference_graph(conn).query("MATCH (n) DETACH DELETE n")
        seeded = _catalog_rows(size)
        seeded[-1] = {**seeded[-1], "productId": "opaque-sku-42"}
        _seed_catalog(conn, seeded)
        shop = _storefront(Services(catalog_repo))
        reference_reads.clear()
        return seeded, shop.list_catalog(), len(reference_reads)

    _, _, reads_for_3 = _cold_list_catalog(3)
    seeded, rows, reads_for_15 = _cold_list_catalog(15)

    assert [row["productId"] for row in rows] == [p["productId"] for p in seeded]
    assert rows[-1] == {
        "productId": "opaque-sku-42", "name": "Widget 015",
        "category": "Accessories", "price": 25.0, "imageUrl": None,
    }
    assert all(
        set(row) == {"productId", "name", "category", "price", "imageUrl"}
        for row in rows
    )
    assert 1 <= reads_for_15 <= 2, (
        f"expected listing 15 products to cost 1-2 reads of `reference`, got "
        f"{reads_for_15} — {_read_summary(reference_reads)}"
    )
    assert reads_for_15 == reads_for_3, (
        f"the catalog read must not scale with the catalog: 3 products cost "
        f"{reads_for_3} reads of `reference`, 15 products cost {reads_for_15} "
        f"— {_read_summary(reference_reads)}"
    )


def test_list_catalog_carries_an_explicit_bound_past_the_delivered_default(
    stocked, conn
):
    """§5.1's S7 row: `services.filter_products` defaults `limit=20` — correct
    for 15 products, **silently wrong at 21**.

    21 is the smallest catalog that can tell the two apart, and the failure it
    guards is invisible: a truncated catalog raises nothing, logs nothing, and
    simply stops offering the last products. Mutation-checked by dropping the
    `limit=CATALOG_LIMIT` argument, which yields 20 rows here and 15 in every
    other catalog test in this file.
    """
    _seed_catalog(conn, _catalog_rows(21))

    rows = stocked.list_catalog()

    assert len(rows) == 21
    assert rows[-1]["productId"] == "widget-021"


def test_the_image_manifest_is_non_empty_against_a_fixture_asset_directory(
    conn, catalog_repo, tmp_path
):
    """§4.7's stated trap: the negative half of AC-11 ("no placeholder element")
    passes unchanged when the manifest is **totally** empty, so the positive
    half has to be asserted on its own.

    Paired with `test_an_unset_storefront_dir_yields_an_empty_manifest` below,
    which is the same code path producing the failure this one rules out.
    """
    _seed_catalog(conn, _catalog_rows(3))
    catalog_repo.ensure_agent(WS, agent_id=AGENT, name="Demo agent", created_at=90)
    root = _assets(tmp_path / "dist", ["widget-001.webp", "widget-003.png"])
    shop = _storefront(Services(catalog_repo), storefront_dir=root)

    manifest = shop.build_image_manifest()

    assert manifest == {
        "widget-001": "/shop/products/widget-001.webp",
        "widget-003": "/shop/products/widget-003.png",
    }
    assert [row["imageUrl"] for row in shop.list_catalog()] == [
        "/shop/products/widget-001.webp", None, "/shop/products/widget-003.png",
    ]


def test_the_manifest_keeps_only_catalog_products_and_known_extensions(
    conn, catalog_repo, tmp_path
):
    """Both halves of §4.7's intersection: an asset with no product never
    becomes a URL, and a file outside `IMAGE_EXTENSIONS` is not an asset."""
    _seed_catalog(conn, _catalog_rows(2))
    catalog_repo.ensure_agent(WS, agent_id=AGENT, name="Demo agent", created_at=90)
    root = _assets(tmp_path / "dist", [
        "widget-001.jpg",        # a real product
        "widget-999.webp",       # an asset for no product in the catalog
        "widget-002.svg",        # a real product, an extension we do not serve
        "index.html",            # the SPA itself, sitting one level up in real life
    ])
    shop = _storefront(Services(catalog_repo), storefront_dir=root)

    assert shop.build_image_manifest() == {
        "widget-001": "/shop/products/widget-001.jpg"
    }


def test_extension_precedence_is_webp_first(conn, catalog_repo, tmp_path):
    _seed_catalog(conn, _catalog_rows(1))
    catalog_repo.ensure_agent(WS, agent_id=AGENT, name="Demo agent", created_at=90)
    root = _assets(tmp_path / "dist", [
        "widget-001.png", "widget-001.jpeg", "widget-001.jpg", "widget-001.webp",
    ])
    shop = _storefront(Services(catalog_repo), storefront_dir=root)

    assert shop.build_image_manifest() == {
        "widget-001": "/shop/products/widget-001.webp"
    }


@pytest.mark.parametrize("layout", ["unset", "no-products-dir", "source-tree"])
def test_a_manifest_with_no_served_assets_is_empty_rather_than_wrong(
    conn, catalog_repo, tmp_path, layout
):
    """The negative control for the non-empty assertion above, in the three
    shapes §4.7 names: no `FALKORCHAT_STOREFRONT_DIR` at all, a build output
    with no `products/`, and the v1.0 defect — assets that exist only in the
    **source tree** while the served directory is `dist/` alone.

    Empty is the correct answer for all three: every `imageUrl` is `null` and
    the client renders its text-only card. What must never happen is a URL for
    a file the server does not serve.
    """
    _seed_catalog(conn, _catalog_rows(2))
    catalog_repo.ensure_agent(WS, agent_id=AGENT, name="Demo agent", created_at=90)
    _assets(tmp_path / "salesperson" / "public", ["widget-001.webp"])
    (tmp_path / "dist").mkdir()
    served = {
        "unset": None,
        "no-products-dir": tmp_path / "dist",
        "source-tree": tmp_path / "dist",
    }[layout]
    shop = _storefront(Services(catalog_repo), storefront_dir=served)

    assert shop.build_image_manifest() == {}
    assert [row["imageUrl"] for row in shop.list_catalog()] == [None, None]


def test_the_manifest_is_built_once_not_per_catalog_call(conn, catalog_repo, tmp_path):
    """§4.7's operational note, asserted rather than documented: the manifest is
    a startup artifact, so an asset dropped in afterwards needs a restart.

    The observable is `list_catalog`, which must not re-list the directory per
    call — and the second half proves the manifest is genuinely rebuildable, so
    "built once" is a policy rather than a one-shot bug.
    """
    _seed_catalog(conn, _catalog_rows(2))
    catalog_repo.ensure_agent(WS, agent_id=AGENT, name="Demo agent", created_at=90)
    root = _assets(tmp_path / "dist", ["widget-001.webp"])
    shop = _storefront(Services(catalog_repo), storefront_dir=root)
    assert shop.build_image_manifest() == {
        "widget-001": "/shop/products/widget-001.webp"
    }

    (root / "products" / "widget-002.webp").write_bytes(b"\x00")

    assert [row["imageUrl"] for row in shop.list_catalog()] == [
        "/shop/products/widget-001.webp", None,
    ]
    # …until the next restart, which is what `build_image_manifest` stands for.
    assert shop.build_image_manifest() == {
        "widget-001": "/shop/products/widget-001.webp",
        "widget-002": "/shop/products/widget-002.webp",
    }


def test_list_catalog_builds_the_manifest_when_nobody_did(conn, catalog_repo, tmp_path):
    """S8 calls `build_image_manifest` from the app's lifespan; a `Storefront`
    built without that step must still serve image URLs rather than silently
    answering `null` for every product."""
    _seed_catalog(conn, _catalog_rows(1))
    catalog_repo.ensure_agent(WS, agent_id=AGENT, name="Demo agent", created_at=90)
    root = _assets(tmp_path / "dist", ["widget-001.webp"])
    shop = _storefront(Services(catalog_repo), storefront_dir=root)

    assert shop.list_catalog()[0]["imageUrl"] == "/shop/products/widget-001.webp"


# ── advance_own_order (§4.6 / §5.2) ──────────────────────────────────────────


def _with_order(shop, name="Ada"):
    """A participant holding one `placed` order for one product."""
    record = shop.join(name, "en")
    ctx = shop.context_for(record.participant_id)
    shop._services.add_cart_item(ctx, product_name="Widget 001", quantity=1)
    order = shop._services.place_order(ctx)
    return record, ctx, order["orderId"]


def test_advance_own_order_walks_the_lifecycle(stocked, conn):
    _seed_catalog(conn, _catalog_rows(2))
    _record, ctx, order_id = _with_order(stocked)

    assert stocked.advance_own_order(ctx, order_id=order_id, transition="fulfill") == {
        "orderId": order_id, "status": "fulfilled",
    }
    assert stocked.advance_own_order(ctx, order_id=order_id, transition="deliver") == {
        "orderId": order_id, "status": "delivered",
    }
    assert stocked.get_state(ctx)["order"]["status"] == "delivered"


def test_advancing_another_participants_order_is_refused_and_changes_nothing(
    stocked, conn
):
    """The gate `services.advance_order` does not have: its CAS is keyed on
    `orderId` alone (graph note §10.2).

    The refusal is asserted **twice over** — the raise, and the victim's order
    still `placed` afterwards — because a wrapper that advanced first and
    checked second would raise here too, and only the second assertion can tell
    the two apart.
    """
    _seed_catalog(conn, _catalog_rows(2))
    _ada, ada_ctx, ada_order = _with_order(stocked, "Ada")
    _bob, bob_ctx, _bob_order = _with_order(stocked, "Bob")

    with pytest.raises(UnknownOrderError):
        stocked.advance_own_order(bob_ctx, order_id=ada_order, transition="cancel")

    assert stocked.get_state(ada_ctx)["order"]["status"] == "placed"


def test_the_ownership_gate_runs_before_the_cas(stocked, conn, monkeypatch):
    """Order of operations, pinned as a call sequence.

    The assertion above proves the *outcome*; this proves the *mechanism* —
    `services.advance_order` is never reached at all for someone else's order.
    Without it, an implementation that advanced and then rolled back would look
    identical from outside, and would still be a window in which another
    participant's order was cancelled.
    """
    _seed_catalog(conn, _catalog_rows(2))
    _ada, _ada_ctx, ada_order = _with_order(stocked, "Ada")
    _bob, bob_ctx, _bob_order = _with_order(stocked, "Bob")
    calls: list[str] = []
    services = stocked._services
    real_gate = services.order_belongs_to_customer
    real_advance = services.advance_order

    def gate(*args, **kwargs):
        calls.append("gate")
        return real_gate(*args, **kwargs)

    def advance(*args, **kwargs):
        calls.append("advance")
        return real_advance(*args, **kwargs)

    monkeypatch.setattr(services, "order_belongs_to_customer", gate)
    monkeypatch.setattr(services, "advance_order", advance)

    with pytest.raises(UnknownOrderError):
        stocked.advance_own_order(bob_ctx, order_id=ada_order, transition="cancel")
    assert calls == ["gate"]

    stocked.advance_own_order(bob_ctx, order_id=_bob_order, transition="fulfill")
    assert calls == ["gate", "gate", "advance"]


def test_an_unknown_order_is_refused_exactly_like_someone_elses(stocked, conn):
    """§5.3 C10: both are `404`, and the client cannot tell them apart — an
    order id is not an oracle for whether an order exists."""
    _seed_catalog(conn, _catalog_rows(2))
    _record, ctx, _order_id = _with_order(stocked)

    with pytest.raises(UnknownOrderError):
        stocked.advance_own_order(ctx, order_id="no-such-order", transition="fulfill")


def test_a_stale_transition_is_refused_with_the_orders_current_status(stocked, conn):
    """The ordinary stale-button outcome: `deliver` pressed before `fulfill`
    landed. `409`, carrying the status the client should repaint from."""
    _seed_catalog(conn, _catalog_rows(2))
    _record, ctx, order_id = _with_order(stocked)

    with pytest.raises(OrderTransitionRefusedError) as exc:
        stocked.advance_own_order(ctx, order_id=order_id, transition="deliver")

    assert exc.value.status == "placed"
    assert exc.value.transition == "deliver"
    assert stocked.get_state(ctx)["order"]["status"] == "placed"


# ── reset mine (§4.8, graph note §4/§7/§12) ──────────────────────────────────


def _thread_message_count(conn, thread_id):
    return _one(
        conn,
        "MATCH (:Thread {threadId: $tid})-[:HEAD]->(:Message)-[:NEXT*0..]->(m:Message) "
        "RETURN count(m)",
        {"tid": thread_id},
    ) or 0


def _dangling_cursors_owned_by(conn, participant_id):
    """Graph note §7 (d), read **participant-scoped** — S7 has no global intake
    stop, so the condition is over cursors owned by the reset participant."""
    return _one(
        conn,
        "MATCH (:User {userId: $pid})-[:HAS_CURSOR]->(rc:ReadCursor) "
        "OPTIONAL MATCH (t:Thread {threadId: rc.threadId}) "
        "WITH rc, t WHERE t IS NULL RETURN count(rc)",
        {"pid": participant_id},
    )


def _stub_run(conn, *, run_id, trigger_msg_id, status="running"):
    """A `WorkflowRun` in the shape the reset sweeps — reached only through
    `TRIGGERED_BY` from a thread message (graph note §4).

    Written raw, as a fixture: `repository.start_run` additionally needs a
    published `WorkflowDefSnapshot` with a START step, and none of that is what
    these tests are about. Same posture as `_seed_catalog` above.
    """
    db.workspace_graph(conn, WS).query(
        "MATCH (m:Message {msgId: $msgId}) "
        "CREATE (:WorkflowRun {runId: $runId, status: $status, "
        "                      defKey: 'salesperson', defVersion: 'v7', "
        "                      startedAt: 1, stepCount: 0, maxSteps: 12, "
        "                      trace: false, ctx: '{}', waitingThreadId: ''})"
        "-[:TRIGGERED_BY]->(m)",
        {"msgId": trigger_msg_id, "runId": run_id, "status": status},
    )


def _run_status(conn, run_id):
    return _one(
        conn, "MATCH (r:WorkflowRun {runId: $rid}) RETURN r.status", {"rid": run_id}
    )


def _busy_participant(shop, conn, name="Ada"):
    """A participant with a transcript, a run trail, a cart and an order — the
    full victim set `reset_participant` is supposed to take."""
    record = shop.join(name, "pt-BR")
    ctx = shop.context_for(record.participant_id)
    services = shop._services
    services.save_profile(ctx, delivery_address="12 Rua das Flores")
    posted = services.post_message(ctx, thread_id=record.thread_id, text="hello")
    services.post_message(
        config.CallContext(ws=WS, actor=AGENT), thread_id=record.thread_id,
        text="how can I help?",
    )
    _stub_run(
        conn, run_id=f"{record.participant_id}-run", trigger_msg_id=posted["msgId"],
        status="done",
    )
    services.add_cart_item(ctx, product_name="Widget 001", quantity=2)
    services.place_order(ctx)
    services.add_cart_item(ctx, product_name="Widget 002", quantity=1)
    return record, ctx


def test_reset_clears_the_participants_state_and_remints_their_thread(stocked, conn):
    _seed_catalog(conn, _catalog_rows(2))
    record, ctx = _busy_participant(stocked, conn)
    pid = record.participant_id
    assert _thread_message_count(conn, record.thread_id) == 2

    result = stocked.reset_participant(record)

    assert set(result) == {"threadId", "language"}
    assert result["language"] == "pt-BR"
    assert result["threadId"] != record.thread_id
    assert result["threadId"].startswith(storefront.THREAD_ID_PREFIX)
    # the transcript, the run trail and the commerce subgraph are gone …
    assert _one(conn, "MATCH (t:Thread {threadId: $t}) RETURN count(t)",
                {"t": record.thread_id}) == 0
    assert _thread_message_count(conn, result["threadId"]) == 0
    assert _run_status(conn, f"{pid}-run") is None
    assert _one(conn, "MATCH (:Customer {customerId: $p})-[:PLACED]->(o:Order) "
                      "RETURN count(o)", {"p": pid}) == 0
    assert _one(conn, "MATCH (c:Cart {customerId: $p}) RETURN count(c)",
                {"p": pid}) == 0
    # The `Customer` anchor itself is back, and deliberately: the profile
    # re-write below runs `upsert_profile`, whose `MERGE` re-creates it. What
    # matters is that it carries the name and nothing else — asserted whole in
    # `test_the_profile_name_is_back_after_a_self_reset_not_an_em_dash`.
    # … while the identity survives, token included (§4.8: reset-mine keeps it)
    assert stocked.resolve_token(_bearer(record)) is not None
    assert _one(conn, "MATCH (u:User {userId: $p}) RETURN u.displayName",
                {"p": pid}) == "Ada"
    assert _one(conn, "MATCH (:Channel {channelId: $c})-[:HAS_THREAD]->(t) "
                      "RETURN t.threadId", {"c": record.channel_id}) == result["threadId"]
    assert stocked.get_state(ctx)["cart"] == {"items": [], "total": 0}
    assert stocked.get_state(ctx)["order"] is None


def test_the_profile_name_is_back_after_a_self_reset_not_an_em_dash(stocked, conn):
    """§2.4's FR-10 parity bar, and the one done-condition of this step that is
    a *second* write rather than a property of the delete.

    The `Customer` node goes with the reset while `User.displayName` survives,
    so without the re-write the profile panel renders an em-dash for a name the
    participant typed on the join screen and never withdrew. `deliveryAddress`
    is asserted `None` in the same breath: it proves the `Customer` really was
    deleted, so the name coming back is a re-write and not a survivor.
    """
    _seed_catalog(conn, _catalog_rows(2))
    record, ctx = _busy_participant(stocked, conn)
    assert stocked.get_state(ctx)["profile"] == {
        "name": "Ada", "deliveryAddress": "12 Rua das Flores",
    }

    stocked.reset_participant(record)

    assert stocked.get_state(ctx)["profile"] == {
        "name": "Ada", "deliveryAddress": None,
    }


def test_a_self_reset_clears_the_dead_turn_latch(stocked, conn):
    """§5.1's S9 row: reset-mine's own clear path, alongside
    `clear_all_turns()`'s — because the transcript the notice refers to is
    gone once the reset commits.

    The latch is set directly (`_mark_turn_failed`) rather than by driving a
    real failing turn: `stocked` carries no trigger, and this test's whole
    subject is the reset's own clear, not how the latch got there — the
    trigger-driven path is `test_a_turn_whose_trigger_raises_is_isolated_and_
    still_clears_the_gate` and `test_a_dead_turn_is_reported_idle_and_failed_
    in_the_same_state_body`'s job.
    """
    _seed_catalog(conn, _catalog_rows(1))
    record, ctx = _busy_participant(stocked, conn)
    stocked._mark_turn_failed(record.participant_id)  # noqa: SLF001
    assert stocked.get_state(ctx)["turn"]["lastTurn"] == "failed"

    stocked.reset_participant(record)

    assert stocked.get_state(ctx)["turn"]["lastTurn"] is None


def test_reset_is_participant_disjoint(stocked, conn):
    _seed_catalog(conn, _catalog_rows(2))
    ada, _ada_ctx = _busy_participant(stocked, conn, "Ada")
    bob, bob_ctx = _busy_participant(stocked, conn, "Bob")
    # Bob's own dead-turn latch is not Ada's reset's business either — §5.1's
    # S9 row: the clear names its own participant, never the whole map.
    stocked._mark_turn_failed(bob.participant_id)  # noqa: SLF001

    stocked.reset_participant(ada)

    assert _thread_message_count(conn, bob.thread_id) == 2
    assert _run_status(conn, f"{bob.participant_id}-run") == "done"
    state = stocked.get_state(bob_ctx)
    assert state["profile"] == {"name": "Bob", "deliveryAddress": "12 Rua das Flores"}
    assert state["order"] is not None
    assert state["cart"]["items"] != []
    assert state["turn"]["lastTurn"] == "failed"
    assert stocked.resolve_token(_bearer(bob)) is not None


def test_resetting_a_non_participant_raises_rather_than_reporting_success(
    stocked, conn
):
    """Zero rows — not a participant, or already deleted (graph note §12)."""
    _seed_catalog(conn, _catalog_rows(1))
    ghost = ParticipantRecord(
        participant_id="p-ghost", display_name="Ghost", language="en",
        channel_id="ch-p-ghost", thread_id="th-p-ghost", joined_at=1,
    )

    with pytest.raises(UnknownParticipantError):
        stocked.reset_participant(ghost)


def test_an_unscoped_participant_is_an_alarm_never_a_success(stocked, conn):
    """`scoped=false` (graph note §4's G2): the participant resolved but their
    own `Channel` did not, so the reset was a **guaranteed no-op**.

    `409` with a machine-readable code, never `200` — "a `200` here is the same
    class of lie v1.0's partial delete told". The transcript is asserted intact
    afterwards: this must be an alarm about a graph that needs repair, not a
    quiet success over a subgraph that was never touched.
    """
    _seed_catalog(conn, _catalog_rows(2))
    record, _ctx = _busy_participant(stocked, conn)
    db.workspace_graph(conn, WS).query(
        "MATCH (c:Channel {channelId: $cid}) SET c.participantId = 'p-someone-else'",
        {"cid": record.channel_id},
    )

    with pytest.raises(UnscopedParticipantError) as exc:
        stocked.reset_participant(record)

    assert exc.value.code == "unscoped_participant"
    assert _thread_message_count(conn, record.thread_id) == 2
    assert _one(conn, "MATCH (c:Customer {customerId: $p}) RETURN count(c)",
                {"p": record.participant_id}) == 1


# ── the quiesce contract (`docs/plans/salesperson-ui-graph.md` §7 (a)–(d)) ───
#
# §7's four conditions **replace** v1.0's "a reset leaves no orphan
# `StepRun`/`TraceEvent`/`Message`", which that note disproved as vacuous: all
# three writes are anchored on nodes the reset deleted, so they create nothing
# post-reset whether quiesce works or not. These four can fail.
#
# They are read **participant-scoped** here, which is what reset-mine is: S7 has
# no global intake stop — that is S10's `reset_all`, and §7 (b)'s "(intake
# stopped)" and (d)'s "after `reset_all`" are worded for it.


def test_the_reset_waits_for_an_in_flight_turn_before_it_deletes(stocked, conn,
                                                                 monkeypatch):
    """§7 **(a)** and **(c)**, asserted at the moment of the delete rather than
    after it.

    (a) asks that the reset "completes only after that turn finishes — assert
    the turn's `WorkflowRun` reached a terminal status *before* the delete".
    So the observer is a spy wrapped around `repository.reset_participant`: it
    reads the run's status and the thread's message count **at the instant the
    single atomic delete is issued**. Asserting them afterwards proves nothing,
    because the delete takes both away.

    (c) — "no turn is silently dropped" — is the message count in the same
    reading: the in-flight turn's reply was written before the delete, so the
    turn ran to completion instead of being cut off mid-flight. Its reply is
    then deleted with the transcript, which is what the participant asked for.

    (b) rides along: the agent's post lands *during* the quiesce window and
    must succeed, not raise `ThreadNotFoundError` against a vanished thread —
    the failure mode §7.3 says quiesce-then-delete exists to prevent.

    Mutation-checked by removing the `_await_quiesce` call from
    `reset_participant`: the spy then fires while the run is still `running`
    and the thread holds one message.
    """
    _seed_catalog(conn, _catalog_rows(2))
    record = stocked.join("Ada", "en")
    pid = record.participant_id
    ctx = stocked.context_for(pid)
    posted = stocked._services.post_message(
        ctx, thread_id=record.thread_id, text="hello"
    )
    _stub_run(conn, run_id="turn-run", trigger_msg_id=posted["msgId"])
    turn_booking = _pin_thinking(stocked, pid)

    at_delete: dict = {}
    real_reset = stocked._repo.reset_participant

    def spy(*args, **kwargs):
        at_delete["runStatus"] = _run_status(conn, "turn-run")
        at_delete["messages"] = _thread_message_count(conn, record.thread_id)
        return real_reset(*args, **kwargs)

    monkeypatch.setattr(stocked._repo, "reset_participant", spy)

    turn_result: dict = {}

    def run_the_turn():
        # Its own connection: this is a worker thread, exactly as S9's executor
        # will be.
        worker = Services(Repository(db.connect()))
        time.sleep(TURN_WORK_S)
        try:
            worker.post_message(
                config.CallContext(ws=WS, actor=AGENT),
                thread_id=record.thread_id, text="how can I help?",
            )
            turn_result["posted"] = True
        except Exception as exc:  # noqa: BLE001 — recorded, asserted below
            turn_result["error"] = exc
        db.workspace_graph(db.connect(), WS).query(
            "MATCH (r:WorkflowRun {runId: 'turn-run'}) SET r.status = 'done'"
        )
        stocked.release_turn(pid, turn_booking)
        # The instant the turn left the map — i.e. the earliest moment
        # `_await_quiesce` could possibly have stopped waiting.
        turn_result["finished_at"] = time.monotonic()

    turn = threading.Thread(target=run_the_turn)
    turn.start()
    try:
        # Bounded well above this storefront's own 5 s `quiesce_s`, so a genuine
        # refusal still surfaces as one rather than as a hang (S7-3).
        outcome = _call_bounded(stocked.reset_participant, record, seconds=10)
    finally:
        turn.join(timeout=5)
    result = outcome["result"]
    finished_at = turn_result["finished_at"]

    # **The wait itself**, as a pure ordering of two instants in this process —
    # no duration, so no margin to be flaky about. Together the two lines say
    # the reset was issued *before* the turn ended and returned *after* it,
    # which is what "it waited" means:
    #
    #   worker:  ├─ in flight ─────────── finished_at ──┤
    #   reset:      started_at ─────── (blocked) ─── returned_at
    #
    # Both of the reset's instants are read on the daemon thread that makes the
    # call, which is what makes "no margin" literally true: a `started_at` read
    # out here on the main thread would be stamped before that thread was even
    # scheduled, and the first line below would then pass on the strength of the
    # thread-start skew alone (Pass 8, S8-1).
    #
    # Every other assertion in this test is also satisfied by the ordering in
    # which the worker finishes *before* the reset starts — nothing to wait for
    # — which the review ran and found kept all four of them green (Pass 7,
    # S7-1). The first line is the one that ordering reddens; the second is what
    # a reset that never waits reddens.
    assert outcome["started_at"] < finished_at, (
        "the reset was not issued while the turn was in flight — this test "
        "proves nothing about waiting"
    )
    assert outcome["returned_at"] >= finished_at, "the reset did not wait"

    assert turn_result["posted"] is True            # (b): the post did not raise
    assert "error" not in turn_result, turn_result.get("error")
    assert at_delete["runStatus"] == "done"         # (a): terminal before the delete
    assert at_delete["messages"] == 2               # (c): the reply was written
    assert result["threadId"] != record.thread_id


def test_a_quiesce_timeout_changes_nothing_and_leaves_the_turn_running(
    stocked, conn
):
    """§4.8/§7.1's `503` branch, and the other half of §7 (c): the client saw a
    refusal, so nothing was dropped.

    `quiesce_s=0` is the whole waiting budget, so an in-flight turn cannot
    drain and the reset must refuse. "Changes nothing" is asserted as the
    node count before and after — the reset is one atomic query, so a partial
    delete is not a shape the graph can be left in, but a reset that ran *at
    all* would show up here.
    """
    _seed_catalog(conn, _catalog_rows(2))
    shop = _storefront(stocked._services, quiesce_s=0)
    record, ctx = _busy_participant(shop, conn)
    _pin_thinking(shop, record.participant_id)
    before = _one(conn, "MATCH (n) RETURN count(n)")

    # `_call_bounded`, not a bare call: a zero budget has nothing to wait for,
    # so this must refuse at once — and a broken `_await_quiesce` deadline would
    # otherwise block on a wall clock this test does not control, which is a CI
    # job timeout naming no test at all rather than a failure (Pass 7, S7-3).
    with pytest.raises(QuiesceTimeoutError):
        _call_bounded(shop.reset_participant, record)

    assert _one(conn, "MATCH (n) RETURN count(n)") == before
    assert _thread_message_count(conn, record.thread_id) == 2
    assert shop.turn_in_flight(record.participant_id) is True
    assert shop.get_state(ctx)["order"] is not None


def test_an_idle_participant_is_not_made_to_wait(stocked, conn):
    """The control for the two above: with no turn in flight the reset does not
    consult the clock at all, so a `quiesce_s=0` storefront resets normally.

    Without this, "the reset waits" and "the reset refuses" would both be
    satisfied by a `reset_participant` that always refused.
    """
    _seed_catalog(conn, _catalog_rows(2))
    shop = _storefront(stocked._services, quiesce_s=0)
    record, _ctx = _busy_participant(shop, conn)

    # Bounded by `IMMEDIATE_S`, and this is the one bounded call that writes —
    # read that constant's comment before changing it.
    outcome = _call_bounded(shop.reset_participant, record)

    assert outcome["result"]["threadId"] != record.thread_id


# ── cancelling a *queued* turn, in front of the wait (§4.8, S9b) ─────────────
#
# The wait above is unchanged and these tests do not weaken it: what S9b adds
# runs **before** it, and everything the cancel cannot reach still arrives
# there. Two properties have to be separated everywhere below, because a turn
# that was cancelled and a turn that ran and cleared up leave the *same* empty
# map entry behind:
#
#   * the `Future` is `cancelled()` — the executor will never run that item;
#   * `trigger.ran` never names its `msg_id` — the workflow layer never saw it.
#
# An assertion on the map alone proves neither, which is exactly why the
# rejected design (drop the entry, skip the wait) would pass one written that
# way. `_queue_holding_trigger` exists to make the second reading possible.


def test_a_queued_turn_is_cancelled_and_never_reaches_the_workflow_layer(services):
    """The positive case, end to end at the map/queue layer.

    `turn_workers=1` with the first turn parked on the worker makes the second
    one's work item provably `PENDING`, which is the only state
    `Future.cancel()` acts on. The three readings are the three things the
    design claims: the handle really is on the map entry (*one map, not two*,
    §5.1's S9 row), the future is really cancelled, and the slot is cleared —
    in that order, since the clear is conditional on the cancel.

    `trigger.ran` is what makes this more than a map assertion: the cancelled
    turn never reached `maybe_trigger` at all, so no LLM call was spent and no
    reply will land against a thread reset-mine is about to delete.
    """
    trigger, entered, gate, _later, _later_gate = _queue_holding_trigger()
    shop = _storefront(services, trigger=trigger, turn_workers=1)
    held = _enqueue(
        shop, shop.context_for("p-hold"), _participant("p-hold"),
        _posted("m-hold", "th-hold"),
    )
    assert entered.wait(timeout=IMMEDIATE_S), "the holding turn never ran"

    queued = _enqueue(
        shop, shop.context_for("p-ada"), _participant("p-ada"),
        _posted("m-ada", "th-ada"),
    )
    assert shop.turn_state("p-ada").state == TURN_QUEUED
    assert shop.turn_state("p-ada").future is queued

    assert shop._cancel_queued_turn("p-ada") is True

    assert queued.cancelled() is True
    assert shop.turn_state("p-ada") == IDLE_TURN

    gate.set()
    held.result(timeout=IMMEDIATE_S)
    assert trigger.ran == ["m-hold"]


def test_a_cancel_that_loses_the_race_to_the_worker_leaves_the_slot_standing(
    services, monkeypatch
):
    """**The race the ordering exists for**, forced rather than hoped for: the
    future leaves `PENDING` *between* the map read and the `cancel()` call.

    `Future.cancel` is wrapped so that, on the one future this test is about,
    the worker is released and observed entering that very turn before the real
    `cancel` runs. Nothing in the code under test is altered — the wrapper
    delegates, so the `False` below is CPython's own answer for a running work
    item, not a stub's.

    What must then hold is the whole of the ordering rule: the entry is **still
    there**, still owned by that booking, and reset-mine falls through to
    `_await_quiesce`. A cancel that cleared the slot first would report idle
    under a live turn and let the delete race it — the state §4.8 calls
    *actively wrong*, and the one the map-only spelling of this file's other
    quiesce tests cannot see.
    """
    trigger, entered, gate, later, later_gate = _queue_holding_trigger()
    later_gate.clear()  # hold the racing turn *inside* the trigger
    shop = _storefront(services, trigger=trigger, turn_workers=1)
    held = _enqueue(
        shop, shop.context_for("p-hold"), _participant("p-hold"),
        _posted("m-hold", "th-hold"),
    )
    assert entered.wait(timeout=IMMEDIATE_S), "the holding turn never ran"
    queued = _enqueue(
        shop, shop.context_for("p-ada"), _participant("p-ada"),
        _posted("m-ada", "th-ada"),
    )
    assert shop.turn_state("p-ada").state == TURN_QUEUED

    real_cancel = Future.cancel

    def racing_cancel(self):  # noqa: ANN001, ANN202
        if self is queued:
            gate.set()
            assert later.wait(timeout=IMMEDIATE_S), "the racing turn never ran"
        return real_cancel(self)

    monkeypatch.setattr(Future, "cancel", racing_cancel)

    assert shop._cancel_queued_turn("p-ada") is False

    assert queued.cancelled() is False
    assert shop.turn_in_flight("p-ada") is True
    assert shop.turn_state("p-ada").state == TURN_THINKING

    later_gate.set()
    held.result(timeout=IMMEDIATE_S)
    queued.result(timeout=IMMEDIATE_S)
    assert trigger.ran == ["m-hold", "m-ada"]
    assert shop.turn_in_flight("p-ada") is False


def test_a_late_attach_never_resurrects_a_finished_turns_entry(services):
    """`enqueue_turn` attaches the future *after* `submit` returns, by which
    time the worker may have run the whole turn and cleared the slot.

    An unconditional attach there would write the entry back — a `queued` turn
    nobody will ever clear, which `409`-refuses that participant for the life of
    the process and answers every reset-mine of theirs `503`. That is P17-3's
    shape through a door the release rule does not cover, since nothing was
    refused. The attach is therefore ownership-checked like every other write on
    this class, and this drives the finished-turn ordering directly rather than
    waiting for it to happen.
    """
    trigger = _RecordingTrigger()
    shop = _storefront(services, trigger=trigger)
    booking = shop.reserve_turn("p-ada")
    future = shop.enqueue_turn(
        shop.context_for("p-ada"), _participant("p-ada"),
        _posted("m-1", "th-ada"), booking,
    )
    _drain(shop, future, trigger)
    assert shop.turn_state("p-ada") == IDLE_TURN

    assert shop._attach_turn_future("p-ada", booking, future) is False

    assert shop.turn_state("p-ada") == IDLE_TURN
    assert shop.turn_in_flight("p-ada") is False


def test_the_cancel_reaches_the_live_booking_and_never_the_orphan(services):
    """The `clear_all_turns()` arrangement, on the cancel path.

    Reset-all empties the map under running workers and the participant can post
    again, so two of their bookings exist and the map holds one slot. The rule
    is the same one the `finally` and the release already obey: the entry
    belongs to the booking that owns it, so the stale future must not overwrite
    the fresh one's handle, and the cancel must reach the **fresh** turn.

    The orphan is not abandoned by losing its slot — it was accepted, so it runs
    to completion and only its bookkeeping is skipped, which the last two lines
    read off the trigger rather than off the map.
    """
    trigger, entered, gate, _later, _later_gate = _queue_holding_trigger()
    shop = _storefront(services, trigger=trigger, turn_workers=1)
    held = _enqueue(
        shop, shop.context_for("p-hold"), _participant("p-hold"),
        _posted("m-hold", "th-hold"),
    )
    assert entered.wait(timeout=IMMEDIATE_S), "the holding turn never ran"

    old = shop.reserve_turn("p-ada")
    orphan = shop.enqueue_turn(
        shop.context_for("p-ada"), _participant("p-ada"),
        _posted("m-orphan", "th-ada"), old,
    )
    shop.clear_all_turns()
    fresh = shop.reserve_turn("p-ada")
    live = shop.enqueue_turn(
        shop.context_for("p-ada"), _participant("p-ada"),
        _posted("m-live", "th-ada"), fresh,
    )

    assert shop.turn_state("p-ada").booking == fresh
    assert shop.turn_state("p-ada").future is live
    assert shop._attach_turn_future("p-ada", old, orphan) is False

    assert shop._cancel_queued_turn("p-ada") is True
    assert live.cancelled() is True
    assert orphan.cancelled() is False

    gate.set()
    held.result(timeout=IMMEDIATE_S)
    orphan.result(timeout=IMMEDIATE_S)
    assert trigger.ran == ["m-hold", "m-orphan"]


def test_a_reset_cancels_a_queued_turn_where_it_used_to_refuse(stocked, conn):
    """§4.8's availability half, at the reset itself: `quiesce_s=0` — the whole
    budget — and the reset **succeeds**.

    This is the arrangement S7 could only refuse. The wait is unchanged and
    still has nothing to wait for at zero budget; what changed is that the turn
    is gone from the queue before the wait is asked, so there is nothing left in
    flight. Its neighbour
    `test_a_quiesce_timeout_changes_nothing_and_leaves_the_turn_running` is the
    control that keeps this from reading as "reset-mine stopped refusing": a
    turn already on a worker still produces the `503`.

    The `trigger.ran` reading is what separates this from the design §4.8
    rejected — dropping the map entry as a stand-in produces the same `200` and
    the same empty slot, with the turn still queued behind it.
    """
    _seed_catalog(conn, _catalog_rows(2))
    trigger, entered, gate, _later, _later_gate = _queue_holding_trigger()
    shop = _storefront(
        stocked._services, quiesce_s=0, turn_workers=1, trigger=trigger
    )
    holder = shop.join("Grace", "en")
    holder_ctx = shop.context_for(holder.participant_id)
    held_post = shop._services.post_message(
        holder_ctx, thread_id=holder.thread_id, text="hold the worker"
    )
    held = _enqueue(shop, holder_ctx, holder, held_post)
    assert entered.wait(timeout=IMMEDIATE_S), "the holding turn never ran"

    record, ctx = _busy_participant(shop, conn)
    pid = record.participant_id
    mine = shop._services.post_message(ctx, thread_id=record.thread_id, text="mine")
    queued = _enqueue(shop, ctx, record, mine)
    assert shop.turn_state(pid).state == TURN_QUEUED

    outcome = _call_bounded(shop.reset_participant, record)

    assert queued.cancelled() is True
    assert shop.turn_in_flight(pid) is False
    assert outcome["result"]["threadId"] != record.thread_id

    gate.set()
    held.result(timeout=IMMEDIATE_S)
    assert trigger.ran == [held_post["msgId"]]


def test_the_reset_cancels_only_the_resetting_participants_turn(stocked, conn):
    """The cancel is keyed on **the participant being reset**, and nothing else.

    Two queued turns behind one busy worker, one reset. A lookup keyed on a
    fixed or wrong id satisfies every map assertion the other tests make —
    something *was* cancelled and a slot *is* empty — and fails here on both
    sides at once: the resetting participant's turn survives (so `quiesce_s=0`
    refuses instead of resetting) and the bystander's is destroyed.

    The two `msgId`s are the ones FalkorDB minted for the two posts, so the
    reading below cannot pass on a value this test invented.
    """
    trigger, entered, gate, _later, _later_gate = _queue_holding_trigger()
    shop = _storefront(
        stocked._services, quiesce_s=0, turn_workers=1, trigger=trigger
    )
    holder = shop.join("Grace", "en")
    holder_ctx = shop.context_for(holder.participant_id)
    held_post = shop._services.post_message(
        holder_ctx, thread_id=holder.thread_id, text="hold the worker"
    )
    held = _enqueue(shop, holder_ctx, holder, held_post)
    assert entered.wait(timeout=IMMEDIATE_S), "the holding turn never ran"

    mine_rec = shop.join("Ada", "en")
    mine_ctx = shop.context_for(mine_rec.participant_id)
    mine_post = shop._services.post_message(
        mine_ctx, thread_id=mine_rec.thread_id, text="reset me"
    )
    mine = _enqueue(shop, mine_ctx, mine_rec, mine_post)

    other_rec = shop.join("Bob", "en")
    other_ctx = shop.context_for(other_rec.participant_id)
    other_post = shop._services.post_message(
        other_ctx, thread_id=other_rec.thread_id, text="leave me alone"
    )
    other = _enqueue(shop, other_ctx, other_rec, other_post)

    outcome = _call_bounded(shop.reset_participant, mine_rec)

    assert mine.cancelled() is True
    assert other.cancelled() is False
    assert shop.turn_state(other_rec.participant_id).state == TURN_QUEUED
    assert outcome["result"]["threadId"] != mine_rec.thread_id

    gate.set()
    held.result(timeout=IMMEDIATE_S)
    other.result(timeout=IMMEDIATE_S)
    assert trigger.ran == [held_post["msgId"], other_post["msgId"]]


def test_the_reset_leaves_no_dangling_cursor_owned_by_the_participant(stocked, conn):
    """§7 **(d)**, participant-scoped — the direct test for F3's one real orphan
    class (`advance_cursor` `MERGE`s on the *member*, not the thread, so it can
    mint a `ReadCursor` naming a thread that no longer exists).

    The second participant is the false-positive control: a reset that swept
    every cursor in the workspace would satisfy (d) just as well, and would be
    a different, worse defect.
    """
    _seed_catalog(conn, _catalog_rows(2))
    ada, _ada_ctx = _busy_participant(stocked, conn, "Ada")
    bob, _bob_ctx = _busy_participant(stocked, conn, "Bob")
    repo = stocked._services._repo
    for member, thread in ((ada, ada.thread_id), (bob, bob.thread_id)):
        repo.advance_cursor(
            WS, me_id=member.participant_id, thread_id=thread,
            cursor_id=f"{member.participant_id}:{thread}", now=500, now_msg_id="x",
        )
    repo.advance_cursor(
        WS, me_id=ada.participant_id, thread_id="th-long-gone",
        cursor_id=f"{ada.participant_id}:th-long-gone", now=501, now_msg_id="x",
    )
    assert _dangling_cursors_owned_by(conn, ada.participant_id) == 1

    stocked.reset_participant(ada)

    assert _dangling_cursors_owned_by(conn, ada.participant_id) == 0
    assert repo.get_cursor(
        WS, cursor_id=f"{bob.participant_id}:{bob.thread_id}"
    ) is not None


# ── F8 — a socket timeout means *unknown*, never "nothing changed" ───────────


class _Timeout(redis_exceptions.TimeoutError):
    """A FalkorDB socket timeout, in the exact class `db.connect()` raises."""


def test_a_socket_timeout_on_the_reset_is_unknown_with_a_fresh_state_read(
    stocked, conn, monkeypatch
):
    """§4.8 F8 / `docs/QUERIES.md` §18.7, first ordering.

    The module's `TIMEOUT` applies to reads only, so a slow reset is never
    truncated server-side; if one crosses `FALKORDB_SOCKET_TIMEOUT` the client
    raises **while the server commits the delete**. So this is `504
    reset_state_unknown` after re-reading state — never the quiesce `503`,
    whose whole meaning is "nothing changed".

    The re-read is asserted to be a *real* read of the graph, not a
    placeholder: this stub times out without deleting anything, so the state it
    reports still carries the order and the cart.

    `calls == 1` is the application half of §4.8's stated premise that nothing
    retries a reset. (The library half is out of this test's reach by
    construction, which is why the premise is written down.)
    """
    _seed_catalog(conn, _catalog_rows(2))
    record, _ctx = _busy_participant(stocked, conn)
    calls: list[tuple] = []

    def timing_out(*args, **kwargs):
        calls.append((args, kwargs))
        raise _Timeout("Timeout reading from socket")

    monkeypatch.setattr(stocked._repo, "reset_participant", timing_out)

    with pytest.raises(ResetStateUnknownError) as exc:
        stocked.reset_participant(record)

    assert exc.value.code == "reset_state_unknown"
    assert not isinstance(exc.value, QuiesceTimeoutError)
    assert len(calls) == 1
    assert exc.value.state is not None
    assert set(exc.value.state) == {"profile", "cart", "order", "turn"}
    assert exc.value.state["order"] is not None
    assert exc.value.state["cart"]["items"] != []


def test_a_socket_timeout_on_the_re_read_too_is_still_unknown_never_a_500(
    stocked, conn, monkeypatch
):
    """§4.8 F8, second ordering — **the likelier fault, not the exotic one**.

    The re-read is another query against the same graph, and FalkorDB
    serialises writes per graph, so the stalled reset that produced the first
    timeout is precisely what stalls the re-read for another
    `FALKORDB_SOCKET_TIMEOUT`. A fake that times out on the reset and *succeeds*
    on the re-read exercises only the easier half, which is why both orderings
    are separately named.

    The contract: still `504 reset_state_unknown`, simply with no state body.
    The participant-facing meaning is identical either way — *unknown*, never
    "nothing changed" — and a bare `TimeoutError` escaping as a `500` is the
    failure this rules out.
    """
    _seed_catalog(conn, _catalog_rows(2))
    record, _ctx = _busy_participant(stocked, conn)

    def timing_out(*args, **kwargs):
        raise _Timeout("Timeout reading from socket")

    monkeypatch.setattr(stocked._repo, "reset_participant", timing_out)
    monkeypatch.setattr(stocked._repo, "get_profile", timing_out)

    with pytest.raises(ResetStateUnknownError) as exc:
        stocked.reset_participant(record)

    assert exc.value.code == "reset_state_unknown"
    assert exc.value.state is None
    assert exc.value.participant_id == record.participant_id
    # …and it is the storefront's own refusal, not the raw client error.
    assert isinstance(exc.value, StorefrontError)
    assert not isinstance(exc.value, redis_exceptions.TimeoutError)


def test_a_runtime_error_on_the_re_read_is_also_unknown_never_a_500(
    stocked, conn, monkeypatch
):
    """`get_state`'s **other** call site, and the gap `## Pass 22` (P22-4)
    found: `_reset_state_unknown`'s re-read used to catch only
    `redis_exceptions.TimeoutError`, so a `RuntimeError` reached through
    `get_state` on this leg escaped as a bare `500` instead of F8's `504` —
    breaking the exact promise `_reset_state_unknown`'s own docstring makes
    ("still a `504`, never a `500`"). Nothing raises `RuntimeError` through
    `get_state` today (the module's raises-guard pins that class to
    `enqueue_turn` alone), so this fakes the shape the guard would otherwise
    let through unnoticed, the same way the sibling test above fakes a second
    `TimeoutError` rather than waiting for a real socket to misbehave.

    The contract is identical to the sibling test's: still `504
    reset_state_unknown`, with no state body — a `RuntimeError` escaping
    uncaught is the failure this rules out.
    """
    _seed_catalog(conn, _catalog_rows(2))
    record, _ctx = _busy_participant(stocked, conn)

    def timing_out(*args, **kwargs):
        raise _Timeout("Timeout reading from socket")

    def raises_runtime_error(*args, **kwargs):
        raise RuntimeError("no actor on the state read")

    monkeypatch.setattr(stocked._repo, "reset_participant", timing_out)
    monkeypatch.setattr(stocked._repo, "get_profile", raises_runtime_error)

    with pytest.raises(ResetStateUnknownError) as exc:
        stocked.reset_participant(record)

    assert exc.value.code == "reset_state_unknown"
    assert exc.value.state is None
    assert exc.value.participant_id == record.participant_id
    # …and it is `ResetStateUnknownError` itself, not the raw `RuntimeError`
    # passing through unmapped — `StorefrontError` is a `RuntimeError`
    # subclass, so the discriminator is the type, not the class hierarchy.
    assert isinstance(exc.value, ResetStateUnknownError)
    assert "no actor on the state read" not in str(exc.value)


def test_the_two_reset_failures_are_different_exceptions(stocked, conn, monkeypatch):
    """The pairing F8 exists to enforce, stated as one assertion: a quiesce
    timeout and a socket timeout are **not** the same refusal.

    v1.8 read F8's "client" as the browser and mapped both to `503 … nothing
    changed`. They are answered by two disjoint types here, so S8 cannot map
    them to one status by accident.
    """
    _seed_catalog(conn, _catalog_rows(2))
    record, _ctx = _busy_participant(stocked, conn)

    assert not issubclass(ResetStateUnknownError, QuiesceTimeoutError)
    assert not issubclass(QuiesceTimeoutError, ResetStateUnknownError)

    busy = _storefront(stocked._services, quiesce_s=0)
    _pin_thinking(busy, record.participant_id)
    # Bounded for the same reason as above (S7-3): fail, never hang.
    with pytest.raises(QuiesceTimeoutError):
        _call_bounded(busy.reset_participant, record)

    monkeypatch.setattr(
        stocked._repo, "reset_participant",
        lambda *a, **k: (_ for _ in ()).throw(_Timeout("boom")),
    )
    with pytest.raises(ResetStateUnknownError):
        stocked.reset_participant(record)
