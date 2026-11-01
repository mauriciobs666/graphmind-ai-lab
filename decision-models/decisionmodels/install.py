"""`driver install`: put the agent definitions and the guard into a campaign's `exchange/` and
record the declared expectations (plan `docs/plans/typed-decision-models4.md` §3.1, §3.4, §3.5;
plan6 `typed-decision-models6.md` §3.2 "User settings", §3.3).

**Sources** are `agents/` at the component root: `dm-driver.md`, `dm-generator.md`, `dm-labeler.md`,
`spawn-generator.txt`, `spawn-labeler.txt`, `spawn-driver.txt`, `guard_hook.py` and `settings.json`
(dataset provenance, hashed into `promptSha256`; not `claude/` or `.claude/agents/`). `--agents-dir`
names another directory with the same files, for the smoke's `maxTurns: 2` copy of `dm-labeler`.
Spawn templates carry no trailing newline: a template is compared byte for byte with the prompt
the harness sends.

**What it does**, all under `Campaign.locked()` (it refuses while a `driver run` holds the lock),
and every refusal comes before any file is written or any event appended:
0. refuses a `claude` below `runner.MIN_CLAUDE_VERSION` (`runner.claude_version_refusal`, plan6
   §3.3 "Version floor");
1. reads `~/.claude/settings.json` and refuses on `runner.forbidden_settings_refusal` (the same
   function `run` applies), then digests it with `runner.user_settings_digests`;
2. checks the sources: each definition's frontmatter has exactly its role's keys and the pinned
   values (`definition_error`, plan6 §3.3 "Definition check"; `hooks` is refused by name);
   `settings.json` wires the guard exactly once with `onFailure: "block"` (`guard_command`) and pins
   `autoContinueAtUsageLimit: false`;
3. byte-copies each source into `exchange/.claude/` (definitions into `agents/`, plus the guard,
   its spawn templates and `settings.json`), then re-reads every installed file and compares its
   sha256 with the source's. An installed file that already differs from its source (a one-byte
   edit), a stray file in `.claude/agents/`, a `.claude/settings.local.json`, or sources that no
   longer hash to what an earlier `expectations` event recorded, refuses: remove the installed
   copy, or start a new campaign;
4. appends an `expectations` event (a fresh one on every run, which is how a refused
   user-settings edit is accepted; `fold` lets only the first event set the model pins and
   declared `instructions`, so a re-run keeps a `resolve` repin): the guard's exact command, per
   role the pins read from the installed definition's frontmatter (`model`, `effort`) with its
   agent type, spawn template and the empty declared `instructions` set, the `driverTemplate`,
   `userSettingsDigests`, `settingsSha256` and
   `promptSha256 {agentDefinition, spawnTemplate, guardHook}` (plan4 §3.7's keys, the driver's
   template included; `rubric` and `requestTemplate` are not install's).

It prints the recorded user-settings key names, never a value.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

from . import runner
from .campaign import Campaign, CampaignRefused, _write_atomic, root_refusal

AGENTS_DIR = Path(__file__).resolve().parents[1] / "agents"
DEFINITIONS = {"driver": "dm-driver.md", "generator": "dm-generator.md",
               "labeler": "dm-labeler.md"}
TEMPLATES = {"generator": "spawn-generator.txt", "labeler": "spawn-labeler.txt",
             "driver": "spawn-driver.txt"}
GUARD = "guard_hook.py"
SETTINGS = "settings.json"
#: Where each source lands under `exchange/.claude/`.
LAYOUT = {**{name: f"agents/{name}" for name in DEFINITIONS.values()},
          TEMPLATES["generator"]: TEMPLATES["generator"],
          TEMPLATES["labeler"]: TEMPLATES["labeler"], GUARD: GUARD, SETTINGS: SETTINGS}
#: The roles whose subagent the audit checks, and the agent type of each.
ROLE_AGENT = {"generator": "dm-generator", "labeler": "dm-labeler"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def frontmatter(text: str) -> dict[str, str]:
    """The `key: value` lines between the leading `---` lines of an agent definition."""
    lines = text.split("\n")
    if not lines or lines[0] != "---" or "---" not in lines[1:]:
        raise CampaignRefused("an agent definition has no frontmatter")
    fields: dict[str, str] = {}
    for line in lines[1:lines.index("---", 1)]:
        key, sep, value = line.partition(":")
        if not sep:
            raise CampaignRefused(f"a frontmatter line is not `key: value`: {line!r}")
        fields[key.strip()] = value.strip()
    return fields


SUBAGENT_KEYS = frozenset({"name", "description", "model", "effort", "tools", "maxTurns",
                           "omitClaudeMd", "permissionMode"})
DRIVER_KEYS = frozenset({"name", "description", "model", "tools", "maxTurns"})
#: Pinned values by role (plan6 §3.3 "Definition check"); `model` and `effort` of a subagent
#: only need to be non-empty (they become the recorded pins), `maxTurns` an integer.
SUBAGENT_PINS = {"tools": "Read, Write", "omitClaudeMd": "true", "permissionMode": "acceptEdits"}
DRIVER_PINS = {"model": "claude-sonnet-5-5", "maxTurns": "3",
               "tools": "Agent(dm-generator, dm-labeler)"}


def definition_error(role: str, fields: Mapping[str, str]) -> str | None:
    """Why a definition's frontmatter is refused (plan6 §3.3 "Definition check"), or `None`: a
    key outside the role's set (`hooks` by name; Claude Code ignores an unknown key, so a misspelt
    one would install silently), a missing key, or a broken pinned value."""
    if "hooks" in fields:
        return "carries `hooks`: the guard is wired in settings only"
    keys, pins = (DRIVER_KEYS, DRIVER_PINS) if role == "driver" else (SUBAGENT_KEYS,
                                                                       SUBAGENT_PINS)
    if unknown := sorted(set(fields) - keys):
        return f"has keys outside its set: {unknown}"
    if missing := sorted(keys - set(fields)):
        return f"lacks {missing}"
    for key, value in pins.items():
        if fields[key] != value:
            return f"`{key}` must be `{value}`"
    if not fields["maxTurns"].isdigit() or int(fields["maxTurns"]) < 1:
        return "`maxTurns` must be a positive integer"
    for key in ("model", "effort"):
        if key in fields and not fields[key]:
            return f"`{key}` is empty"
    return None


def guard_command(settings: str | Mapping) -> str:
    """The guard's exact `command` string from a project `settings.json` (text or parsed): the
    one string the audit compares each `hook_success` with (`Expected.guard_command`). Refuses a
    wiring that is not exactly one `PreToolUse` group, matcher `*`, holding one command handler
    with `"onFailure": "block"`, and any hook event but `PreToolUse` (plan6 §3.3)."""
    try:
        parsed = json.loads(settings) if isinstance(settings, str) else settings
    except json.JSONDecodeError as exc:
        raise CampaignRefused(f"settings.json is not JSON ({exc})") from None
    hooks = parsed.get("hooks") if isinstance(parsed, Mapping) else None
    if not isinstance(hooks, Mapping) or set(hooks) != {"PreToolUse"}:
        raise CampaignRefused("settings.json must wire hooks for PreToolUse only")
    groups = hooks["PreToolUse"]
    if not (isinstance(groups, list) and len(groups) == 1 and isinstance(groups[0], Mapping)
            and groups[0].get("matcher") == "*" and set(groups[0]) == {"matcher", "hooks"}):
        raise CampaignRefused("settings.json must wire exactly one PreToolUse group, matcher `*`")
    handlers = groups[0]["hooks"]
    if not (isinstance(handlers, list) and len(handlers) == 1
            and isinstance(handlers[0], Mapping)):
        raise CampaignRefused("settings.json must wire the guard once")
    handler = handlers[0]
    command = handler.get("command")
    if handler.get("type") != "command" or not isinstance(command, str) or not command:
        raise CampaignRefused("settings.json's guard is not a command hook")
    if handler.get("onFailure") != "block":
        raise CampaignRefused('settings.json\'s guard lacks "onFailure": "block"')
    return command


def _project_settings_errors(text: str) -> None:
    command = guard_command(text)
    if GUARD not in command:
        raise CampaignRefused(f"settings.json's guard command does not run {GUARD}")
    if json.loads(text).get("autoContinueAtUsageLimit") is not False:
        raise CampaignRefused("settings.json must pin autoContinueAtUsageLimit to false")


def _read_sources(agents_dir: Path) -> dict[str, bytes]:
    sources: dict[str, bytes] = {}
    for name in (*LAYOUT, TEMPLATES["driver"]):
        path = agents_dir / name
        if not path.is_file():
            raise CampaignRefused(f"{path} is missing")
        sources[name] = path.read_bytes()
    for role, name in TEMPLATES.items():
        if "{" not in sources[name].decode("utf-8", "replace") or sources[name].endswith(b"\n"):
            raise CampaignRefused(f"{name} must hold its slots and no trailing newline ({role})")
    for role, name in DEFINITIONS.items():
        error = definition_error(role, frontmatter(sources[name].decode("utf-8")))
        if error is not None:
            raise CampaignRefused(f"{name} {error}")
    _project_settings_errors(sources[SETTINGS].decode("utf-8"))
    return sources


def _check_installed(claude: Path, sources: Mapping[str, bytes]) -> None:
    """Refuse an installed file that differs from its source, and anything unexpected."""
    for name, dest in LAYOUT.items():
        installed = claude / dest
        if installed.is_symlink() or (installed.exists() and not installed.is_file()):
            raise CampaignRefused(f"{installed} is not a regular file; remove it")
        if installed.is_file() and sha256(installed.read_bytes()) != sha256(sources[name]):
            raise CampaignRefused(f"{installed} differs from its source (sha256); remove it and "
                                  "re-run `driver install`, or start a new campaign")
    agents = claude / "agents"
    if agents.is_dir():
        stray = sorted(set(os.listdir(agents)) - set(DEFINITIONS.values()))
        if stray:
            raise CampaignRefused(f"{agents} holds files that are not agent definitions: {stray}")
    if (claude / "settings.local.json").exists():
        raise CampaignRefused(f"{claude / 'settings.local.json'} is a second settings channel; "
                              "remove it")


def _prompt_hashes(sources: Mapping[str, bytes]) -> dict[str, dict[str, str] | str]:
    return {"agentDefinition": {role: sha256(sources[name]) for role, name in
                                DEFINITIONS.items()},
            "spawnTemplate": {role: sha256(sources[name]) for role, name in TEMPLATES.items()},
            "guardHook": sha256(sources[GUARD])}


def install(root: Path, *, agents_dir: Path = AGENTS_DIR, settings_path: Path | None = None
            ) -> list[str]:
    """`driver install` (module docstring): the lines to print, or `CampaignRefused`."""
    camp = Campaign(root)
    if not camp.ledger_path.is_file():
        raise CampaignRefused(f"no campaign at {camp.root} (run `driver init`)")
    refusal = root_refusal(camp.root)
    if refusal is not None:
        raise CampaignRefused(refusal)
    with camp.locked():
        version = runner.claude_version_refusal(os.environ)
        if version is not None:
            raise CampaignRefused(version)
        settings = runner.read_user_settings(settings_path or runner.user_settings_path())
        forbidden = runner.forbidden_settings_refusal(settings)
        if forbidden is not None:
            raise CampaignRefused(forbidden)
        digests = runner.user_settings_digests(settings)
        sources = _read_sources(agents_dir)
        claude = camp.exchange_dir / ".claude"
        _check_installed(claude, sources)
        hashes = _prompt_hashes(sources)
        earlier = camp.state().expectations
        if earlier is not None and earlier.get("promptSha256", hashes) != hashes:
            raise CampaignRefused("the sources no longer hash to what this campaign recorded "
                                  "at its first install; start a new campaign")
        for name, dest in LAYOUT.items():
            if not (claude / dest).is_file():
                _write_atomic(claude / dest, sources[name])
        for name, dest in LAYOUT.items():
            if sha256((claude / dest).read_bytes()) != sha256(sources[name]):
                raise CampaignRefused(f"{claude / dest} does not hash to its source after the copy")
        roles = {}
        for role, agent in ROLE_AGENT.items():
            pins = frontmatter((claude / "agents" / DEFINITIONS[role]).read_text("utf-8"))
            roles[role] = {"agentType": agent, "model": pins["model"], "effort": pins["effort"],
                           "spawnTemplate": sources[TEMPLATES[role]].decode("utf-8"),
                           "instructions": []}
        command = guard_command((claude / SETTINGS).read_text("utf-8"))
        camp.record_expectations(
            guard_command=command, roles=roles,
            driver_template=sources[TEMPLATES["driver"]].decode("utf-8"),
            user_settings_digests=digests, prompt_sha256=hashes,
            settings_sha256=sha256(sources[SETTINGS]))
    keys = ", ".join(f"`{key}`" for key in sorted(digests)) or "(none)"
    return [f"installed the agent definitions and guard into {claude}",
            f"guard command: {command}",
            f"recorded user-settings keys: {keys}",
            "a later change to a recorded key refuses `driver run` until `driver install` is "
            "run again"]
