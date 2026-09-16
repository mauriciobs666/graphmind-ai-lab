#!/usr/bin/env python3
"""scripts/s6_walkthrough.py — a human-invoked review aid for S6's FR-19 Step 6 (the stakeholder's
own, binding, non-delegable pass — `docs/plans/small-model-benchmarking-s6-spec.md` §2.6 item 3,
§5 Step 6; this component's own status in `model-bench/AGENTS.md`).

**Never imported by any run-path module (FR-23) and never invoked automatically** — same category
of tool as `scripts/refresh_golden.py`, not part of `modelbench`'s own runtime import graph.

**Read-only.** This script writes nothing — not `conversations.jsonl`, not `PROVENANCE.md`, not
any other pack file. In particular it MUST NEVER touch `provenance.verifiedBy`/`verifiedAt`:
filling those two fields is exclusively the stakeholder's own action (§2.6's own binding ruling —
"no row is filled by anyone other than the stakeholder"), never something a tool automates, even
under instruction. If a future reader is tempted to "improve" this script into an auto-writer of
`verifiedBy`, that is exactly the substitution §2.6 rules out; don't.

What it does: loads the real `tool-caller-shop-assistant` pack, builds one fresh `ShopEnvironment`
per script (state carried across that script's own turns, matching `tools/sim.py`'s per-
conversation-instance contract — the same pattern the agent pre-check used,
`model-bench/docs/reviews/small-model-benchmarking-s6-precheck.md`), and for every turn prints the
scripted customer utterance plus its `expect` block exactly as authored. For a `toolRequired: true`
turn it additionally dispatches the scripted `tool`/`args` against the real, live environment and
prints the real return value, so the reader can see directly whether the `expect` block's own
assertions (`finalReplyMustContain`/`finalReplyMustNotContain`) are true of the real environment's
actual behavior. A restraint turn (`toolRequired` false or absent) is printed with a note that no
tool call is expected — nothing is dispatched or guessed for it.

Usage (from `model-bench/`, with the venv the component's own `setup.sh` creates):

    .venv/bin/python scripts/s6_walkthrough.py
    .venv/bin/python scripts/s6_walkthrough.py --pack packs/tool-caller-shop-assistant  # (default)
    .venv/bin/python scripts/s6_walkthrough.py --script A-03    # one script only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from modelbench.convo import Conversation
from modelbench.packs import load_pack

_DEFAULT_PACK_ROOT = Path(__file__).resolve().parents[1] / "packs" / "tool-caller-shop-assistant"


def _format_expect(expect: dict[str, Any]) -> str:
    """`expect`'s fields, read straight off the JSONL row — never re-derived or paraphrased."""
    tool_required = expect.get("toolRequired", False)
    if not tool_required:
        return "    expect: toolRequired=false (no tool call expected)"

    lines = [
        f"    expect: toolRequired=true tool={expect.get('tool')!r} args={expect.get('args')!r}"
    ]
    if "finalReplyMustContain" in expect:
        lines.append(f"    expect: finalReplyMustContain={expect['finalReplyMustContain']!r}")
    if "finalReplyMustNotContain" in expect:
        lines.append(f"    expect: finalReplyMustNotContain={expect['finalReplyMustNotContain']!r}")
    return "\n".join(lines)


def print_script(conversation: Conversation, environment: Any) -> None:
    """Prints one script's header, then every turn: the scripted utterance, its `expect` block
    verbatim, and — for a `toolRequired: true` turn only — the real, live dispatch result for that
    exact `tool`/`args` pair. One fresh `environment` per call, state carried across this script's
    own turns (the caller builds and owns it, so a `--script` single-script run and a full walk
    share the exact same per-script construction)."""
    print("=" * 88)
    print(
        f"SCRIPT {conversation.scriptId}  (shape {conversation.shape}, "
        f"{len(conversation.turns)} turns)"
    )
    if conversation.description:
        print(f"  {conversation.description}")
    print("=" * 88)

    for turn in conversation.turns:
        print(f"\n[turn {turn.seq}] customer: {turn.user}")
        print(_format_expect(dict(turn.expect)))

        expect = turn.expect
        if expect.get("toolRequired"):
            tool_name = expect.get("tool")
            args = expect.get("args", {})
            result = environment.dispatch(tool_name, args)
            print(f"    LIVE dispatch({tool_name!r}, {args!r}) -> {result!r}")
        else:
            print("    (restraint turn — no tool dispatched)")

    print()


def run_walkthrough(pack_root: Path, only_script: str | None) -> None:
    pack = load_pack(pack_root)
    entrypoint_name = pack.manifest["tools"]["entrypoint"]
    tool_module = pack.load_tool_module()
    build_environment = getattr(tool_module, entrypoint_name)

    scripts: list[Conversation] = list(pack.iter_scripts())
    if only_script is not None:
        scripts = [s for s in scripts if s.scriptId == only_script]
        if not scripts:
            print(f"s6_walkthrough.py: no script {only_script!r} in {pack_root}", file=sys.stderr)
            raise SystemExit(1)

    for conversation in scripts:
        environment = build_environment()  # fresh instance per script, per sim.py's own contract
        print_script(conversation, environment)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "S6 stakeholder-review walkthrough: drives every conversations.jsonl script's "
            "toolRequired turns against the REAL, live tool environment and prints the actual "
            "result beside each turn's scripted expect block. Read-only — never writes "
            "conversations.jsonl, PROVENANCE.md, or provenance.verifiedBy."
        )
    )
    parser.add_argument(
        "--pack",
        type=Path,
        default=_DEFAULT_PACK_ROOT,
        help=f"pack directory (default: {_DEFAULT_PACK_ROOT})",
    )
    parser.add_argument(
        "--script",
        default=None,
        help="print only this scriptId (default: all scripts, in file order)",
    )
    args = parser.parse_args(argv)
    run_walkthrough(args.pack, args.script)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
