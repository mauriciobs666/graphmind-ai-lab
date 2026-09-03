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
import re
import threading
import time
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
    """**Mutation-tested.** Give `resolve_token` a cache-first branch — return
    `self._records[participant_id]` when present, before the graph read — and
    this test goes red: the record is cached by `join` and by the successful
    resolve below, so the deleted participant keeps authenticating out of stale
    memory. The graph read is the only thing that makes the reset real.
    """
    record = seeded.join("Ada", "en")
    assert seeded.resolve_token(_bearer(record)) is not None  # before
    assert record.participant_id in seeded.cached_ids()  # and it *is* cached

    repo.reset_all_participants(WS)

    assert seeded.resolve_token(_bearer(record)) is None  # after
    # The cache is evicted on the way past, so it cannot outlive the registry.
    assert record.participant_id not in seeded.cached_ids()


def test_a_rebuilt_storefront_resolves_a_token_minted_by_the_previous_instance(
    seeded, services
):
    """Restart survival (§4.3) — **mutation-tested**. Make the registry
    authoritative in-process (drop the graph read and answer `resolve_token`
    from `self._records` alone) and this goes red: the second instance's map is
    empty, so every participant would be bounced to a fresh `participantId` and
    lose their cart and order, because `customerId == participantId`.

    The second `Storefront` is built on its own `Repository` over its own
    connection and its cache is asserted empty *before* it answers, so the only
    thing the two instances share is `ws:test` itself.
    """
    record = seeded.join("Ada", "pt-BR")
    bearer = _bearer(record)

    restarted = _storefront(Services(Repository(db.connect())))

    assert restarted is not seeded
    assert restarted.cached_ids() == frozenset()
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


def test_resolving_refreshes_the_cache_so_lookup_never_serves_a_stale_record(
    seeded, repo
):
    """The *other* half of the read-through cache: `resolve_token`'s `_cache_put`
    is what keeps `lookup` fresh, and `lookup` is the only reader there is.

    Written because the refresh was unpinned: deleting that one line passed all
    2439 tests in the repository (`docs/reviews/salesperson-ui-impl.md` Pass 6,
    S6-1), while `lookup` — the accessor S7/S9 are pointed at — went on serving
    the join-time record forever. The observer here is therefore `lookup`, not
    `resolve_token`: the test above already covers the read path, and it stays
    green under the deletion this one catches.

    `lookup` populates on a miss but never re-reads a hit, so nothing else in
    the module can refresh the entry.
    """
    record = seeded.join("Ada", "en")
    assert seeded.resolve_token(_bearer(record)) is not None
    assert seeded.lookup(record.participant_id).language == "en"  # populated, fresh

    repo.set_participant_record(WS, participant_id=record.participant_id, language="es")

    # The cache is still holding the old value — `lookup` alone cannot notice.
    assert seeded.lookup(record.participant_id).language == "en"
    # …until the next authenticated request refreshes it as a side effect.
    assert seeded.resolve_token(_bearer(record)).language == "es"
    assert seeded.lookup(record.participant_id).language == "es"


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


def test_lookup_reads_through_on_a_cache_miss(seeded, services):
    record = seeded.join("Ada", "en")
    fresh = _storefront(Services(Repository(db.connect())))
    assert fresh.cached_ids() == frozenset()

    found = fresh.lookup(record.participant_id)

    assert found is not None
    assert found.display_name == "Ada"
    assert found.thread_id == record.thread_id
    assert fresh.cached_ids() == frozenset({record.participant_id})
    assert fresh.lookup("p-nobody") is None


def test_forget_and_forget_all_drop_cached_records(seeded):
    ada = seeded.join("Ada", "en")
    bob = seeded.join("Bob", "en")
    assert seeded.cached_ids() == frozenset({ada.participant_id, bob.participant_id})

    seeded.forget(ada.participant_id)
    assert seeded.cached_ids() == frozenset({bob.participant_id})

    seeded.forget_all()
    assert seeded.cached_ids() == frozenset()
    # Forgetting is a cache operation, not a logout: the credential still works.
    assert seeded.resolve_token(_bearer(bob)) is not None


def test_the_cache_never_holds_a_raw_token(seeded):
    """The raw credential exists in the participant's browser and in the record
    `join` hands back once — not in the registry map."""
    record = seeded.join("Ada", "en")

    assert record.token is not None
    assert seeded.lookup(record.participant_id).token is None
    assert record.without_token().token is None


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


def test_turn_state_defaults_to_idle(seeded):
    assert seeded.turn_state("p-nobody") == IDLE_TURN
    assert seeded.turn_state("p-nobody").as_payload() == {
        "state": TURN_IDLE, "queuePosition": 0,
    }
    assert seeded.turn_in_flight("p-nobody") is False


def test_a_queued_turn_carries_its_position_and_gates_a_second_post(seeded):
    seeded.set_turn_state("p-a", TURN_QUEUED, queue_position=2)

    assert seeded.turn_state("p-a").as_payload() == {
        "state": TURN_QUEUED, "queuePosition": 2,
    }
    assert seeded.turn_in_flight("p-a") is True

    seeded.set_turn_state("p-a", TURN_THINKING)
    assert seeded.turn_state("p-a").as_payload() == {
        "state": TURN_THINKING, "queuePosition": 0,
    }
    assert seeded.turn_in_flight("p-a") is True


def test_returning_to_idle_clears_the_entry(seeded):
    seeded.set_turn_state("p-a", TURN_THINKING)
    assert seeded.set_turn_state("p-a", TURN_IDLE) == IDLE_TURN
    assert seeded.turn_in_flight("p-a") is False

    seeded.set_turn_state("p-a", TURN_QUEUED, queue_position=1)
    seeded.clear_turn("p-a")
    assert seeded.turn_in_flight("p-a") is False


def test_turn_state_is_per_participant(seeded):
    seeded.set_turn_state("p-a", TURN_THINKING)
    seeded.set_turn_state("p-b", TURN_QUEUED, queue_position=1)

    assert seeded.turn_in_flight("p-a") is True
    assert seeded.turn_in_flight("p-b") is True
    assert seeded.turn_in_flight("p-c") is False

    seeded.clear_all_turns()
    assert seeded.turn_in_flight("p-a") is False
    assert seeded.turn_in_flight("p-b") is False


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
    stocked.set_turn_state(record.participant_id, TURN_QUEUED, queue_position=2)

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
    assert state["turn"] == {"state": TURN_QUEUED, "queuePosition": 2}


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
        "turn": {"state": TURN_IDLE, "queuePosition": 0},
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
    stocked.set_turn_state(ada.participant_id, TURN_THINKING)

    bob_state = stocked.get_state(bob_ctx)

    assert bob_state["profile"] == {"name": "Bob", "deliveryAddress": None}
    assert bob_state["cart"] == {"items": [], "total": 0}
    assert bob_state["order"] is None
    assert bob_state["turn"] == {"state": TURN_IDLE, "queuePosition": 0}
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


def test_reset_is_participant_disjoint(stocked, conn):
    _seed_catalog(conn, _catalog_rows(2))
    ada, _ada_ctx = _busy_participant(stocked, conn, "Ada")
    bob, bob_ctx = _busy_participant(stocked, conn, "Bob")

    stocked.reset_participant(ada)

    assert _thread_message_count(conn, bob.thread_id) == 2
    assert _run_status(conn, f"{bob.participant_id}-run") == "done"
    state = stocked.get_state(bob_ctx)
    assert state["profile"] == {"name": "Bob", "deliveryAddress": "12 Rua das Flores"}
    assert state["order"] is not None
    assert state["cart"]["items"] != []
    assert stocked.resolve_token(_bearer(bob)) is not None


def test_reset_refreshes_the_cached_record_so_lookup_never_serves_a_dead_thread(
    stocked, conn
):
    """The cached record's `threadId` is stale the instant the reset returns,
    and `lookup` never re-reads a hit (S6) — so the reset has to write the new
    one through itself. Without that, S9's worker would post into a thread that
    no longer exists and raise `ThreadNotFoundError`.
    """
    _seed_catalog(conn, _catalog_rows(2))
    record, _ctx = _busy_participant(stocked, conn)
    assert stocked.lookup(record.participant_id).thread_id == record.thread_id

    result = stocked.reset_participant(record)

    cached = stocked.lookup(record.participant_id)
    assert cached.thread_id == result["threadId"]
    assert cached.display_name == "Ada"
    assert cached.token is None


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


def test_a_reset_that_finds_no_participant_evicts_the_cached_record(stocked, conn):
    """The eviction on the zero-row branch, pinned.

    Written because it was not: removing `self._cache_drop(participant_id)` from
    that branch left all 79 tests green, while the success path's `_cache_put`
    reddened its own test — the asymmetry is the finding (Pass 7, S7-2). The
    cache must not outlive the registry entry it mirrors, and this is the branch
    that says the entry is gone.

    The participant is deleted out from under a cached record, which is what
    `reset_all` does to everyone (S10) — so the record is live in the cache and
    the reset then finds nothing, in that order.
    """
    _seed_catalog(conn, _catalog_rows(2))
    record, _ctx = _busy_participant(stocked, conn)
    pid = record.participant_id
    assert pid in stocked.cached_ids()
    db.workspace_graph(conn, WS).query(
        "MATCH (u:User {userId: $pid}) DETACH DELETE u", {"pid": pid}
    )

    with pytest.raises(UnknownParticipantError):
        stocked.reset_participant(record)

    assert pid not in stocked.cached_ids()


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
    stocked.set_turn_state(pid, TURN_THINKING)

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
        stocked.clear_turn(pid)
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
    shop.set_turn_state(record.participant_id, TURN_THINKING)
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


@pytest.mark.parametrize("reread", ["succeeds", "times-out-too"])
def test_a_reset_that_times_out_evicts_the_cached_record(
    stocked, conn, monkeypatch, reread
):
    """The eviction on the F8/`504` branch — **the path where it matters most**,
    and the one it was unpinned on (Pass 7, S7-2).

    `_reset_state_unknown`'s own docstring gives the reason: the delete **may
    have committed**, so the cached `threadId` is exactly as likely to be dead
    as alive, and *unknown* is the one state in which serving a cached record is
    a guess. Removing the `_cache_drop` left 79 tests green.

    `resolve_token` refreshes on the participant's next authenticated request,
    which bounds the damage — but the exposure is an S9 worker calling `lookup`
    **between** the failed reset and that request, and this is the branch where
    that window opens. Both orderings are parametrized because the eviction
    happens before the re-read, so a re-read that also times out must not skip
    it.
    """
    _seed_catalog(conn, _catalog_rows(2))
    record, _ctx = _busy_participant(stocked, conn)
    pid = record.participant_id
    assert pid in stocked.cached_ids()

    def timing_out(*args, **kwargs):
        raise _Timeout("Timeout reading from socket")

    monkeypatch.setattr(stocked._repo, "reset_participant", timing_out)
    if reread == "times-out-too":
        monkeypatch.setattr(stocked._repo, "get_profile", timing_out)

    with pytest.raises(ResetStateUnknownError):
        stocked.reset_participant(record)

    assert pid not in stocked.cached_ids()


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
    busy.set_turn_state(record.participant_id, TURN_THINKING)
    # Bounded for the same reason as above (S7-3): fail, never hang.
    with pytest.raises(QuiesceTimeoutError):
        _call_bounded(busy.reset_participant, record)

    monkeypatch.setattr(
        stocked._repo, "reset_participant",
        lambda *a, **k: (_ for _ in ()).throw(_Timeout("boom")),
    )
    with pytest.raises(ResetStateUnknownError):
        stocked.reset_participant(record)
