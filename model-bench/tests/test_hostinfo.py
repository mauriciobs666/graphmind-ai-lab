"""`modelbench.hostinfo` — the `host.json` schema, its writer/reader, and the attestation
staleness trip-wire's pure decision function (plan §3.4.4, §3.4.4a, §3.4.5 point 3).

This suite never opens a real socket. `attest()`'s `client` argument is a small stand-in exposing
only `.probe()`, the same injected-network seam `tests/test_lmstudio.py` uses one layer down at
`LMStudio(opener=...)`. `check_attestation_staleness` takes no client at all — it is a pure
function over an already-read `host.json` and values a (not-yet-built) `run` unit would have
already observed.

Two `-m live` tests at the bottom are R-1's probe (plan §4 S2, §6 R-1) — written and never run by
this suite: agents are not authorised to load a model in LM Studio.
"""

from __future__ import annotations

import json

import pytest

from modelbench import fingerprint
from modelbench.hostinfo import (
    ATTESTED_FIELD_NAMES,
    STALE_MESSAGE,
    AttestProbeFailed,
    HostInfoError,
    attest,
    check_attestation_staleness,
    host_info_path,
    read_host_info,
    validate_host_info,
    write_host_info,
)
from modelbench.lmstudio import LMStudio

# The plan's own literal (§3.4.4) — a check-input taken from the spec's text, not from a fixture
# this implementation wrote for itself.
PLAN_LITERAL_HOST_JSON = {
    "schemaVersion": 1,
    "apiBaseUrl": "http://localhost:1234",
    "attested": {
        "lmStudioAppVersion": "0.3.31",
        "kvCacheSetting": "f16",
        "hostRamGb": 16,
        "otherResidentWorkloads": ["docker: falkordb-dev", "windows desktop session"],
    },
    "attestedAt": "2026-09-02T14:05:00Z",
    "observedAtAttestation": {"residencySource": "lmstudio-api-v0"},
}


class _FakeClient:
    """Stands in for `LMStudio` — only `probe()` is ever called by `attest()`."""

    def __init__(self, probe_result: str) -> None:
        self._probe_result = probe_result

    def probe(self) -> str:
        return self._probe_result


# --- validate_host_info ---------------------------------------------------------------------


def test_the_plans_own_literal_host_json_validates() -> None:
    """The guard's declared reach checked against the spec's own words, not against a fixture
    this implementation invented for itself."""
    assert validate_host_info(PLAN_LITERAL_HOST_JSON) == []


def test_validate_host_info_rejects_a_non_object() -> None:
    assert validate_host_info([1, 2, 3]) != []
    assert validate_host_info(None) != []
    assert validate_host_info("host.json") != []


@pytest.mark.parametrize("key", ["schemaVersion", "apiBaseUrl", "attested", "attestedAt",
                                  "observedAtAttestation"])
def test_validate_host_info_rejects_a_missing_top_level_key(key) -> None:
    d = json.loads(json.dumps(PLAN_LITERAL_HOST_JSON))
    del d[key]
    problems = validate_host_info(d)
    assert problems != []
    assert any(key in p for p in problems)


def test_validate_host_info_rejects_an_unknown_schema_version() -> None:
    d = json.loads(json.dumps(PLAN_LITERAL_HOST_JSON))
    d["schemaVersion"] = 2
    assert any("schemaVersion" in p for p in validate_host_info(d))


def test_validate_host_info_rejects_an_empty_api_base_url() -> None:
    d = json.loads(json.dumps(PLAN_LITERAL_HOST_JSON))
    d["apiBaseUrl"] = ""
    assert any("apiBaseUrl" in p for p in validate_host_info(d))


@pytest.mark.parametrize("field", ATTESTED_FIELD_NAMES)
def test_validate_host_info_rejects_a_missing_attested_field(field) -> None:
    d = json.loads(json.dumps(PLAN_LITERAL_HOST_JSON))
    del d["attested"][field]
    problems = validate_host_info(d)
    assert any(f"attested.{field}" in p for p in problems)


@pytest.mark.parametrize("field", ATTESTED_FIELD_NAMES)
def test_validate_host_info_rejects_a_null_attested_field(field) -> None:
    d = json.loads(json.dumps(PLAN_LITERAL_HOST_JSON))
    d["attested"][field] = None
    problems = validate_host_info(d)
    assert any(f"attested.{field}" in p and "null" in p for p in problems)


def test_validate_host_info_rejects_a_non_numeric_host_ram_gb() -> None:
    d = json.loads(json.dumps(PLAN_LITERAL_HOST_JSON))
    d["attested"]["hostRamGb"] = "16"
    assert any("hostRamGb" in p for p in validate_host_info(d))


def test_validate_host_info_rejects_a_boolean_host_ram_gb() -> None:
    """`isinstance(True, int)` is `True` in Python — the same trap `Fingerprint.validate()` guards
    against for `benchSchemaVersion`, checked here for the same reason."""
    d = json.loads(json.dumps(PLAN_LITERAL_HOST_JSON))
    d["attested"]["hostRamGb"] = True
    assert any("hostRamGb" in p for p in validate_host_info(d))


def test_validate_host_info_rejects_a_non_list_other_resident_workloads() -> None:
    d = json.loads(json.dumps(PLAN_LITERAL_HOST_JSON))
    d["attested"]["otherResidentWorkloads"] = "docker: falkordb-dev"
    assert any("otherResidentWorkloads" in p for p in validate_host_info(d))


def test_validate_host_info_accepts_an_empty_other_resident_workloads() -> None:
    """`[]` is the correct, informative clean-box answer — not itself a validation problem."""
    d = json.loads(json.dumps(PLAN_LITERAL_HOST_JSON))
    d["attested"]["otherResidentWorkloads"] = []
    assert validate_host_info(d) == []


def test_validate_host_info_rejects_observed_at_attestation_missing_residency_source() -> None:
    d = json.loads(json.dumps(PLAN_LITERAL_HOST_JSON))
    d["observedAtAttestation"] = {}
    assert any("residencySource" in p for p in validate_host_info(d))


def test_validate_host_info_rejects_an_empty_residency_source() -> None:
    d = json.loads(json.dumps(PLAN_LITERAL_HOST_JSON))
    d["observedAtAttestation"]["residencySource"] = ""
    assert any("residencySource" in p for p in validate_host_info(d))


def test_validate_host_info_accepts_backfilled_runtime_keys() -> None:
    """A record the trip-wire has already back-filled (plan §3.4.4a step 6) is still valid — the
    schema names no upper bound on what `observedAtAttestation` may additionally carry."""
    d = json.loads(json.dumps(PLAN_LITERAL_HOST_JSON))
    d["observedAtAttestation"]["runtimeName"] = "llama.cpp"
    d["observedAtAttestation"]["runtimeVersion"] = "1.52.0"
    d["observedAtAttestation"]["runtimeObservedAt"] = "2026-09-03T10:00:05Z"
    assert validate_host_info(d) == []


# --- P13-4: validate_host_info must agree with the fingerprint's own attested-field tiering ---
#
# `lmStudioAppVersion`, `kvCacheSetting` and `hostRamGb` are `_NONEMPTY` in `fingerprint.py`;
# pre-fix, `validate_host_info` checked only presence and non-`null`, so `attest` could write a
# `host.json` that `store()` refuses twenty minutes later, at the end of a whole run (review Pass
# 13, P13-4). Rather than hand-writing a second set of "these three are non-empty" rules — the
# exact shape that drifted the first time — the fix drives `validate_host_info`'s *behavior*
# against `fingerprint.REQUIRED_BY_SCHEMA`'s own live tiers, so a tier change on that side reddens
# this test rather than silently drifting one file over.


def test_validate_host_info_rejects_the_reviews_own_m4a_repro() -> None:
    """Review Pass 13, Appendix M.4/A's exact input — verified pre-fix to return `[]`, clean."""
    d = json.loads(json.dumps(PLAN_LITERAL_HOST_JSON))
    d["attested"] = {
        "lmStudioAppVersion": "",
        "kvCacheSetting": "",
        "hostRamGb": 0,
        "otherResidentWorkloads": [],
    }
    problems = validate_host_info(d)
    assert problems != []
    assert any("lmStudioAppVersion" in p for p in problems)
    assert any("kvCacheSetting" in p for p in problems)
    assert any("hostRamGb" in p for p in problems)


def test_validate_host_info_agrees_with_the_fingerprints_own_attested_field_tiering() -> None:
    """P13-4's required assertion: not two hand-written sets compared to each other, but
    `validate_host_info`'s actual behavior driven against `fingerprint.py`'s own live tiers. Any
    attested field the fingerprint tiers `_NONEMPTY` must be refused empty here; any field it
    tiers `_PRESENT` must be accepted empty (`otherResidentWorkloads`, unaffected)."""
    specs = fingerprint.REQUIRED_BY_SCHEMA[1]["model:chat"]
    for name in ATTESTED_FIELD_NAMES:
        d = json.loads(json.dumps(PLAN_LITERAL_HOST_JSON))
        d["attested"][name] = [] if name == "otherResidentWorkloads" else (
            0 if name == "hostRamGb" else ""
        )
        problems = validate_host_info(d)
        flagged = any(f"attested.{name}" in p for p in problems)
        is_nonempty_tier = specs[name].tier == "nonempty"
        assert flagged == is_nonempty_tier, (name, specs[name].tier, problems)


def test_attested_field_tiers_agree_between_chat_and_embeddings_profiles() -> None:
    """The test above reads tiers from the single `"model:chat"` profile. This is the checkable
    claim that makes that a safe reference rather than an assumption: none of the four attested
    fields is among `model:embeddings`'s four forbidden runtime/sampling fields
    (`fingerprint.py`'s `_EMBEDDINGS_HAVE_NO`), so both profiles must require them at the same
    tier — if a future edit ever moved one of these four into that forbidden set, this reddens
    rather than leaving the test above silently reading the wrong profile."""
    chat = fingerprint.REQUIRED_BY_SCHEMA[1]["model:chat"]
    embeddings = fingerprint.REQUIRED_BY_SCHEMA[1]["model:embeddings"]
    for name in ATTESTED_FIELD_NAMES:
        assert name in embeddings, name
        assert chat[name].tier == embeddings[name].tier, name


def test_validate_host_info_rejects_an_unexpected_attested_key() -> None:
    """P13-12 — §3.4.4: the `attested` block "is exactly the four FR-7 fields". The CLI's own
    `--set` already closes this route (`_parse_set_flags` rejects an unrecognized key), so this is
    only reachable by hand-editing `host.json` — still worth refusing on read."""
    d = json.loads(json.dumps(PLAN_LITERAL_HOST_JSON))
    d["attested"]["surpriseField"] = "x"
    assert any("surpriseField" in p for p in validate_host_info(d))


# --- write_host_info / read_host_info -------------------------------------------------------


def test_write_then_read_round_trips(tmp_root) -> None:
    write_host_info(tmp_root, PLAN_LITERAL_HOST_JSON)
    assert read_host_info(tmp_root) == PLAN_LITERAL_HOST_JSON


def test_write_host_info_creates_the_root_directory(tmp_root) -> None:
    root = tmp_root / "nested" / "dir"
    path = write_host_info(root, PLAN_LITERAL_HOST_JSON)
    assert path == host_info_path(root)
    assert path.is_file()


def test_read_host_info_raises_when_the_file_is_absent(tmp_root) -> None:
    with pytest.raises(HostInfoError):
        read_host_info(tmp_root)


def test_read_host_info_raises_on_malformed_json(tmp_root) -> None:
    host_info_path(tmp_root).write_text("{not json", encoding="utf-8")
    with pytest.raises(HostInfoError):
        read_host_info(tmp_root)


def test_read_host_info_raises_on_a_schema_invalid_file(tmp_root) -> None:
    d = json.loads(json.dumps(PLAN_LITERAL_HOST_JSON))
    del d["attested"]["kvCacheSetting"]
    host_info_path(tmp_root).write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(HostInfoError, match="kvCacheSetting"):
        read_host_info(tmp_root)


# --- attest() --------------------------------------------------------------------------------

ATTESTED = {
    "lmStudioAppVersion": "0.3.31",
    "kvCacheSetting": "f16",
    "hostRamGb": 16,
    "otherResidentWorkloads": ["docker: falkordb-dev"],
}


def test_attest_writes_a_valid_host_json_on_a_successful_probe(tmp_root) -> None:
    path = attest(
        tmp_root,
        api_base_url="http://localhost:1234",
        attested=ATTESTED,
        client=_FakeClient("api-v0"),
    )
    data = json.loads(path.read_text())
    assert validate_host_info(data) == []
    assert data["attested"] == ATTESTED
    assert data["observedAtAttestation"] == {"residencySource": "lmstudio-api-v0"}
    assert "runtimeName" not in data["observedAtAttestation"]


def test_attest_raises_and_writes_nothing_when_unreachable(tmp_root) -> None:
    with pytest.raises(AttestProbeFailed, match="not reachable at http://localhost:1234"):
        attest(
            tmp_root,
            api_base_url="http://localhost:1234",
            attested=ATTESTED,
            client=_FakeClient("unreachable"),
        )
    assert not host_info_path(tmp_root).exists()


def test_attest_raises_and_writes_nothing_when_only_v1_answers(tmp_root) -> None:
    with pytest.raises(AttestProbeFailed, match="not LM Studio's native /api/v0 catalog"):
        attest(
            tmp_root,
            api_base_url="http://localhost:1234",
            attested=ATTESTED,
            client=_FakeClient("v1-only"),
        )
    assert not host_info_path(tmp_root).exists()


def test_attest_stamps_attested_at_in_utc(tmp_root) -> None:
    import datetime as dt

    path = attest(
        tmp_root,
        api_base_url="http://localhost:1234",
        attested=ATTESTED,
        client=_FakeClient("api-v0"),
        now=lambda: dt.datetime(2026, 9, 2, 14, 5, 0, tzinfo=dt.timezone.utc),
    )
    data = json.loads(path.read_text())
    assert data["attestedAt"] == "2026-09-02T14:05:00Z"


# --- check_attestation_staleness — the trip-wire's outcomes (plan §3.4.5 point 3) ------------
#
# Four tests, one per outcome named in the plan (three stored outcome strings, plus the mismatch
# case that lives inside "compared" as `stale=True` rather than as a fourth string).

FRESH_HOST = json.loads(json.dumps(PLAN_LITERAL_HOST_JSON))  # residencySource only, never observed


def test_trip_wire_is_unavailable_on_an_embeddings_arm_even_with_a_mismatched_host() -> None:
    """The embeddings arm has no `runtime` object to observe at all — the check does not run,
    whatever `host.json` holds, including one that would mismatch on a chat arm."""
    host = json.loads(json.dumps(PLAN_LITERAL_HOST_JSON))
    host["observedAtAttestation"]["runtimeName"] = "llama.cpp"
    host["observedAtAttestation"]["runtimeVersion"] = "1.52.0"
    result = check_attestation_staleness(
        host,
        call_surface="embeddings",
        residency_source="lmstudio-api-v0",
        runtime_name="a-totally-different-runtime",
        runtime_version="9.9.9",
    )
    assert result.outcome == "unavailable"
    assert result.stale is False
    assert result.message is None
    assert result.updated_host is None


def test_trip_wire_is_first_observation_and_backfills_only_the_three_runtime_keys() -> None:
    result = check_attestation_staleness(
        FRESH_HOST,
        call_surface="chat",
        residency_source="lmstudio-api-v0",
        runtime_name="llama.cpp",
        runtime_version="1.52.0",
    )
    assert result.outcome == "first-observation"
    assert result.stale is False
    assert result.message is None
    observed = result.updated_host["observedAtAttestation"]
    assert observed["runtimeName"] == "llama.cpp"
    assert observed["runtimeVersion"] == "1.52.0"
    assert observed["runtimeObservedAt"]
    assert observed["residencySource"] == "lmstudio-api-v0"
    # "touching nothing else — never the attested block, never attestedAt" (§3.4.4)
    assert result.updated_host["attested"] == FRESH_HOST["attested"]
    assert result.updated_host["attestedAt"] == FRESH_HOST["attestedAt"]
    assert validate_host_info(result.updated_host) == []


def test_trip_wire_is_compared_and_not_stale_when_the_runtime_matches() -> None:
    host = json.loads(json.dumps(PLAN_LITERAL_HOST_JSON))
    host["observedAtAttestation"].update(
        runtimeName="llama.cpp", runtimeVersion="1.52.0", runtimeObservedAt="2026-09-02T14:06:00Z"
    )
    result = check_attestation_staleness(
        host,
        call_surface="chat",
        residency_source="lmstudio-api-v0",
        runtime_name="llama.cpp",
        runtime_version="1.52.0",
    )
    assert result.outcome == "compared"
    assert result.stale is False
    assert result.message is None
    assert result.updated_host is None


@pytest.mark.parametrize(
    "changed",
    [
        {"runtime_name": "a-different-runtime"},
        {"runtime_version": "9.9.9"},
        {"residency_source": "lmstudio-v1-only"},
    ],
)
def test_trip_wire_is_compared_and_stale_when_any_one_comparand_differs(changed) -> None:
    host = json.loads(json.dumps(PLAN_LITERAL_HOST_JSON))
    host["observedAtAttestation"].update(
        runtimeName="llama.cpp", runtimeVersion="1.52.0", runtimeObservedAt="2026-09-02T14:06:00Z"
    )
    kwargs = dict(
        call_surface="chat",
        residency_source="lmstudio-api-v0",
        runtime_name="llama.cpp",
        runtime_version="1.52.0",
    )
    kwargs.update(changed)
    result = check_attestation_staleness(host, **kwargs)
    assert result.outcome == "compared"
    assert result.stale is True
    assert result.message == STALE_MESSAGE
    assert result.updated_host is None


def test_trip_wire_first_observation_host_survives_round_trip_and_then_compares_clean() -> None:
    """The whole lifecycle in one test: attest -> first chat run (backfill) -> second chat run
    with the same runtime compares clean, exactly as plan §3.4.4a describes it."""
    first = check_attestation_staleness(
        FRESH_HOST,
        call_surface="chat",
        residency_source="lmstudio-api-v0",
        runtime_name="llama.cpp",
        runtime_version="1.52.0",
    )
    second = check_attestation_staleness(
        first.updated_host,
        call_surface="chat",
        residency_source="lmstudio-api-v0",
        runtime_name="llama.cpp",
        runtime_version="1.52.0",
    )
    assert second.outcome == "compared"
    assert second.stale is False


def test_trip_wire_raises_when_host_has_no_observed_at_attestation() -> None:
    """P13-8 — `write_host_info` never validates and `check_attestation_staleness` did not either,
    so a `host` missing `observedAtAttestation` entirely (reachable by any caller that skips
    `read_host_info`, the only place that guarantees it) previously back-filled straight into an
    `updated_host` that itself fails `validate_host_info` — a file that, once written, makes every
    later run exit `5` until the operator re-attests. Refused here instead, at the precondition
    this function actually depends on, rather than inside the branch it would otherwise corrupt."""
    host = {
        "schemaVersion": 1,
        "apiBaseUrl": "http://localhost:1234",
        "attested": dict(PLAN_LITERAL_HOST_JSON["attested"]),
        "attestedAt": "2026-09-02T14:05:00Z",
        # no "observedAtAttestation" key at all
    }
    with pytest.raises(HostInfoError):
        check_attestation_staleness(
            host,
            call_surface="chat",
            residency_source="lmstudio-api-v0",
            runtime_name="llama.cpp",
            runtime_version="1.52.0",
        )


def test_trip_wire_raises_when_residency_source_is_empty() -> None:
    """The same precondition's other edge — `observedAtAttestation` present but its one required
    field empty, the state `validate_host_info` itself refuses on read."""
    host = json.loads(json.dumps(PLAN_LITERAL_HOST_JSON))
    host["observedAtAttestation"]["residencySource"] = ""
    with pytest.raises(HostInfoError):
        check_attestation_staleness(
            host,
            call_surface="chat",
            residency_source="lmstudio-api-v0",
            runtime_name="llama.cpp",
            runtime_version="1.52.0",
        )


def test_trip_wire_raises_on_a_malformed_host_even_on_the_embeddings_surface() -> None:
    """The precondition is checked before the `call_surface` branch — a malformed `host` is a
    contract violation regardless of which surface asks, not only the surfaces that would
    otherwise touch `observedAtAttestation`."""
    with pytest.raises(HostInfoError):
        check_attestation_staleness(
            {},
            call_surface="embeddings",
            residency_source="lmstudio-api-v0",
            runtime_name="llama.cpp",
            runtime_version="1.52.0",
        )


# --- R-1's probe (plan §4 S2, §6 R-1) — needs a model actually loaded in LM Studio -----------
#
# Agents are not authorised to load a model in LM Studio, so these are written and never run by
# this suite (`-m live`, deselected by default per pyproject.toml's `addopts`). "Either outcome
# satisfies the condition; silence does not" (plan §4 S2) — a human/stakeholder session runs
# `pytest -m live` against a loaded model and records the finding in `docs/HISTORY.md`.


@pytest.mark.live
def test_live_loaded_catalog_entry_reveals_kv_cache_or_load_configuration():
    """R-1's original question: with a model loaded (the warm-up call under JIT auto-load *is*
    the load), does the re-read `GET /api/v0/models` entry expose the KV-cache or load
    configuration? The 2026-09-03 probe saw only `not-loaded` entries. Record the answer in
    `docs/HISTORY.md` either way — if it does, `kvCacheSetting` moves from operator-attested to
    auto-captured (a `fingerprint.py`/`AGENTS.md` change, out of this unit's fences)."""
    client = LMStudio("http://localhost:1234")
    client.chat(
        [{"role": "user", "content": "Hello."}],
        model="<a model already loaded by the operator>",
        temperature=0.0,
        max_tokens=8,
        timeout_s=300.0,
    )
    reloaded = {m.id: m for m in client.catalog() if m.state != "not-loaded"}
    assert reloaded  # the warm-up call above must have left something resident
    # Inspect `reloaded[<model id>]` by hand for a KV-cache/load-config key beyond the plan's
    # known ten (§2.5) and record the finding in docs/HISTORY.md.


@pytest.mark.live
def test_live_loaded_context_length_on_a_loaded_embeddings_model():
    """§3.4.4a's open question: does `loadedContextLength` appear on a loaded *embeddings* model,
    not only a loaded chat model (§2.3's evidence)? Read a loaded embeddings model's catalog
    entry; if the key is absent, the field moves to `model:embeddings`'s `REQUIRED_PRESENT` column
    captured `""` — never out of the required set (plan-gate P4-10) — a `fingerprint.py` change,
    out of this unit's fences."""
    client = LMStudio("http://localhost:1234")
    client.embed(["warm-up"], model="text-embedding-qwen3-embedding-0.6b", timeout_s=300.0)
    entries = {m.id: m for m in client.catalog() if m.state != "not-loaded"}
    assert entries
    assert entries["text-embedding-qwen3-embedding-0.6b"].loaded_context_length is not None
