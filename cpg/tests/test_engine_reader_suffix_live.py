"""P1-21 (g), live half of the path-suffix reads (``docs/plans/connected-code-graph-p1-graph.md`` §5 "Path-suffix reads", §8 P1-21 (g)): the two reads run as
``ccg_engine_<s>`` on a throwaway over planted ``File``/``SkippedFile`` rows. Needs the same environment as ``test_engine_reader_live.py``."""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from loader_helpers import admin_password, skip_or_fail, throwaway_port

from ccg import sysname
from ccg.engine import config
from ccg.engine.reader import CypherReader
from ccg.store import acl
from ccg.store import loader as L

HOST = "127.0.0.1"
SYSTEM = f"p121s{os.getpid() % 1000}"
LIVE = sysname.live_key(SYSTEM)
pytestmark = pytest.mark.live

FILES = [("repo1", "server/falkorchat/api.py"), ("repo2", "server/falkorchat/api.py"), ("repo1", "api.py"), ("repo1", "server/xapi.py")]
SKIPPED = [("repo1", "tests/api.py", "test")]


@pytest.fixture(scope="module")
def suffix_rd(tmp_path_factory):
    port = throwaway_port()
    if port is None:
        skip_or_fail("no throwaway instance: start one with cpg/infra/ccg-falkordb.sh --throwaway up and export CCG_FALKORDB_PORT and CCG_CONFIG_DIR")
    try:
        pw = admin_password()
    except FileNotFoundError as exc:
        skip_or_fail(f"throwaway admin password not readable ({exc})")
    admin = L.connect(HOST, port, username=None, password=pw)
    cred = tmp_path_factory.mktemp("ccg-credentials-p121s")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CCG_CREDENTIALS_DIR", str(cred))
        users = acl.create_users(admin, SYSTEM)
        engine_pw = acl.read_credentials(SYSTEM)["engine"]
    try:
        g = admin.select_graph(LIVE)
        g.query("UNWIND $rows AS r CREATE (:File {repoId: r[0], path: r[1], fileId: 'f:' + r[0] + r[1], unitId: 'u:' + r[0]})", {"rows": [list(r) for r in FILES]})
        g.query("UNWIND $rows AS r CREATE (:SkippedFile {repoId: r[0], path: r[1], unitId: 'u:' + r[0], reason: r[2]})", {"rows": [list(r) for r in SKIPPED]})
        g.query("UNWIND range(1, 22) AS i CREATE (:File {repoId: 'bulk', path: 'd' + toString(i + 100) + '/z.py', fileId: 'b' + toString(i), unitId: 'u'})")
        engine = L.connect(HOST, port, username=sysname.acl_user("engine", SYSTEM), password=engine_pw)
        yield SimpleNamespace(rd=CypherReader(engine, SYSTEM))
    finally:
        for key in list(admin.connection.scan_iter(match=f"ccg_{SYSTEM}*")):
            admin.connection.delete(key)
        for user in users:
            admin.connection.execute_command("ACL", "DELUSER", user)


def _files(w, suffix, repos=("repo1", "repo2")):
    return w.rd.read("file_by_suffix", {"repos": list(repos), "sep": "/", "suffix": suffix})


def test_a_partial_path_matches_every_file_ending_with_it_ordered_by_repo_and_path(suffix_rd):
    rows = _files(suffix_rd, "/falkorchat/api.py")
    assert [(r[0], r[1]) for r in rows] == [("repo1", "server/falkorchat/api.py"), ("repo2", "server/falkorchat/api.py")]


def test_the_match_is_at_a_segment_boundary_and_a_repo_root_file_matches_its_own_name(suffix_rd):
    rows = _files(suffix_rd, "/api.py")
    assert [(r[0], r[1]) for r in rows] == [("repo1", "api.py"), ("repo1", "server/falkorchat/api.py"), ("repo2", "server/falkorchat/api.py")]
    assert all("xapi" not in r[1] for r in rows)


def test_a_skipped_file_is_read_with_its_reason_and_an_unmatched_suffix_reads_nothing(suffix_rd):
    assert suffix_rd.rd.read("skipped_by_suffix", {"repos": ["repo1", "repo2"], "sep": "/", "suffix": "/api.py"}) == [("repo1", "tests/api.py", "u:repo1", "test")]
    assert _files(suffix_rd, "/nothing.py") == []


def test_the_repos_parameter_scopes_the_read(suffix_rd):
    assert {r[0] for r in _files(suffix_rd, "/falkorchat/api.py", repos=("repo2",))} == {"repo2"}
    assert _files(suffix_rd, "/falkorchat/api.py", repos=()) == []


def test_more_than_options_max_matches_return_the_cap_plus_one_lowest_by_repo_and_path(suffix_rd):
    rows = _files(suffix_rd, "/z.py", repos=("bulk",))
    assert len(rows) == config.OPTIONS_MAX + 1 == 21
    assert [r[1] for r in rows] == [f"d{i + 100}/z.py" for i in range(1, 22)]


def test_an_injection_shaped_suffix_is_a_plain_string_that_matches_nothing(suffix_rd):
    assert _files(suffix_rd, "/a' OR 1=1 //") == []
