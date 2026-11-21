"""Hook registration detection: the sync header's `Hook:` state (D-1, S10 acceptance)."""
import json
from dataclasses import replace

import pytest

from coordgraph import drift
from coordgraph.config import Config
from coordgraph.hookreg import hook_status

SCRIPT = "coordination-graph/hooks/post-tool-sync.sh"
PROJECT_CMD = '"${CLAUDE_PROJECT_DIR}"/' + SCRIPT


def entry(cmd=PROJECT_CMD, matcher="Write|Edit|Bash"):
    return {"matcher": matcher, "hooks": [{"type": "command", "command": cmd}]}


def settings(post=None, fail=None):
    hooks = {}
    if post is not None:
        hooks["PostToolUse"] = post
    if fail is not None:
        hooks["PostToolUseFailure"] = fail
    return {"hooks": hooks}


def both(**kw):
    return settings([entry(**kw)], [entry(**kw)])


@pytest.fixture
def root(tmp_path):
    (tmp_path / ".claude").mkdir()
    return tmp_path


@pytest.fixture
def cfg(root):
    return replace(Config.from_env({"COORDGRAPH_REPO_ROOT": str(root)}), hook_disabled=False)


def put(root, cfg_obj, name="settings.json"):
    text = cfg_obj if isinstance(cfg_obj, str) else json.dumps(cfg_obj)
    (root / ".claude" / name).write_text(text)


def test_registered_for_both_events(root, cfg):
    put(root, both())
    assert hook_status(cfg, root) == "registered"


def test_no_settings_files_is_not_registered(root, cfg):
    assert hook_status(cfg, root) == "not registered"


def test_one_event_only_is_not_registered(root, cfg):
    put(root, settings([entry()]))
    assert hook_status(cfg, root) == "not registered"
    put(root, settings(None, [entry()]))
    assert hook_status(cfg, root) == "not registered"


def test_events_may_come_from_different_settings_files(root, cfg):
    put(root, settings([entry()]))
    put(root, settings(None, [entry()]), "settings.local.json")
    assert hook_status(cfg, root) == "registered"


@pytest.mark.parametrize("cmd", [
    "/other-checkout/" + SCRIPT,
    PROJECT_CMD + ".bak",
    "echo " + PROJECT_CMD,
    "other-dir/hooks/post-tool-sync.sh",
    '"${CLAUDE_PROJECT_DIR}"/coordination-graph/../elsewhere/hooks/post-tool-sync.sh',
])
def test_a_command_that_is_not_this_script_is_not_registered(root, cfg, cmd):
    put(root, both(cmd=cmd))
    assert hook_status(cfg, root) == "not registered"


@pytest.mark.parametrize("cmd", [
    PROJECT_CMD,
    "$CLAUDE_PROJECT_DIR/" + SCRIPT,
    SCRIPT,
    "./" + SCRIPT,
    "bash " + SCRIPT,
])
def test_the_documented_and_equivalent_command_forms_are_registered(root, cfg, cmd):
    put(root, both(cmd=cmd))
    assert hook_status(cfg, root) == "registered"


def test_the_absolute_path_of_this_checkout_is_registered(root, cfg):
    put(root, both(cmd=str(root / SCRIPT)))
    assert hook_status(cfg, root) == "registered"


@pytest.mark.parametrize("matcher,expected", [
    ("", "registered"), ("*", "registered"), (".*", "registered"),
    ("(Write|Edit|Bash)", "registered"), ("^(Write|Edit|Bash)$", "registered"),
    ("Write|Edit|Bash|Read", "registered"),
    ("Write|Edit", "not registered"), ("Bash", "not registered"),
    ("Write | Edit | Bash", "not registered"),
    ("(unclosed", "not registered"), ("Wri.|Edi.|Bas.", "not registered"),
])
def test_matcher_must_cover_write_edit_and_bash(root, cfg, matcher, expected):
    put(root, both(matcher=matcher))
    assert hook_status(cfg, root) == expected


def test_matcher_tokens_are_unioned_across_entries_of_one_event(root, cfg):
    ev = [entry(matcher="Write|Edit"), entry(matcher="Bash")]
    put(root, settings(ev, ev))
    assert hook_status(cfg, root) == "registered"


@pytest.mark.parametrize("text", ["{not json", "", "[]", '{"hooks": []}', '{"hooks": {"PostToolUse": "x"}}',
                                  '{"hooks": {"PostToolUse": [1], "PostToolUseFailure": [{"hooks": 3}]}}'])
def test_unreadable_or_garbled_settings_are_not_registered_and_never_raise(root, cfg, text):
    put(root, text)
    assert hook_status(cfg, root) == "not registered"


def test_a_garbled_file_does_not_hide_a_good_one(root, cfg):
    put(root, "{garbage")
    put(root, both(), "settings.local.json")
    assert hook_status(cfg, root) == "registered"


def test_the_kill_switch_overrides_registration(root, cfg):
    put(root, both())
    assert hook_status(replace(cfg, hook_disabled=True), root) == "disabled"


def test_the_disabled_file_overrides_registration_even_when_unregistered(root, cfg, tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "disabled").touch()
    assert hook_status(replace(cfg, state_dir=state), root) == "disabled"


def test_header_prints_the_state_verbatim():
    rep = drift.DriftReport(0, None, None, [])
    for state in ("registered", "not registered", "disabled"):
        assert drift.render_header(rep, state).splitlines()[0].endswith(f" · Hook: {state}")
