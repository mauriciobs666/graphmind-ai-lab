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
import inspect
import re
from pathlib import Path

import pytest

from falkorchat import config, db
from falkorchat.repository import Repository
from falkorchat.services import Services
from falkorchat.storefront import (
    IDLE_TURN,
    TURN_IDLE,
    TURN_QUEUED,
    TURN_THINKING,
    DemoNotSeededError,
    ParticipantRecord,
    Storefront,
    hash_token,
    parse_bearer,
)

WS = "test"
AGENT = "assistant"
LOCALES = ("en", "pt-BR", "es")


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


def test_the_token_comparison_is_constant_time():
    """`hmac.compare_digest`, not `==` (§4.3) — asserted **statically**, and
    deliberately so.

    Constant-time comparison has no observable behaviour: `==` passes every
    functional test in this file. A timing measurement would be the only dynamic
    alternative and is not reliable in a unit suite. So this reads the source of
    `resolve_token` and asserts the call is there and that the stored hash is
    never compared with an operator — a tripwire against a future "simplify",
    not evidence that the comparison is fast.
    """
    source = inspect.getsource(Storefront.resolve_token)
    body = source.split('"""', 2)[-1]  # past the docstring

    assert "hmac.compare_digest(stored_hash, hash_token(token))" in body
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
_PACKAGE_DIR = Path(__file__).resolve().parents[1] / "falkorchat"


def test_config_reads_exactly_the_documented_storefront_env_vars():
    """The seven S6 names, spelled once each. Pins `_PRESENTER_KEY` under the
    `FALKORCHAT_STOREFRONT_` prefix, which is how §5.1's S6 row lists it (the
    same elision as `_DIR`/`_TURN_WORKERS`, against `FALKORCHAT_THREAD_LIMIT`
    written out in full because it does *not* take the prefix)."""
    read = set(re.findall(r'"(FALKORCHAT_[A-Z_]+)"', _CONFIG_SOURCE))

    assert {name for name in read if "STOREFRONT" in name} == {
        "FALKORCHAT_STOREFRONT_ENABLED",
        "FALKORCHAT_STOREFRONT_DIR",
        "FALKORCHAT_STOREFRONT_PRESENTER_KEY",
        "FALKORCHAT_STOREFRONT_TURN_WORKERS",
        "FALKORCHAT_STOREFRONT_QUIESCE_S",
        "FALKORCHAT_STOREFRONT_LOCALES",
    }
    assert "FALKORCHAT_THREAD_LIMIT" in read


def test_no_second_workspace_variable_exists_anywhere_in_the_package():
    """§4.9 move 2: the storefront's workspace **is** `config.WS_ID`. B3 was only
    possible because two variables could disagree; with one, the
    misconfiguration is not expressible. This is the tripwire against
    reintroducing `FALKORCHAT_DEMO_WS` by reflex."""
    offenders = [
        str(path.relative_to(_PACKAGE_DIR))
        for path in _PACKAGE_DIR.rglob("*.py")
        if "FALKORCHAT_DEMO_WS" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


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
