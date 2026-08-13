# QA testing techniques — this lab's environment

> **On-demand knowledge base for `qa-engineer`.** Environment/tooling techniques discovered
> driving real black-box QA passes in this lab, not test-plan content — those live in the
> component's own `docs/test-plans/`. Consult when a pass needs one of these mechanics.
>
> Origin: distilled 2026-08-11 from the `qa-engineer` agent's learnings inbox via
> `agent-maintenance` skill §5.

## This WSL2 box has no native browser-automation stack — Windows Chrome + raw CDP over the mirrored-network localhost is the fallback for visual/interactive checks

`playwright`/`selenium` aren't installed for the WSL Python, and there's no Linux-side Node
(`node`/`npx` resolve only to the Windows install). Working path: launch Windows Chrome headless
with remote debugging (`"/mnt/c/Program Files/Google/Chrome/Application/chrome.exe" --headless
--disable-gpu --remote-debugging-port=<port> --user-data-dir="C:\Windows\Temp\<profile>" <url>`),
enumerate targets from WSL with `curl http://localhost:<port>/json` (mirrored networking makes the
port visible cross-side), then drive it with a small script run via the **Windows** `node.exe` (not
WSL Node, which doesn't exist) that opens the page's `webSocketDebuggerUrl` and sends
`Runtime.evaluate`/`Page.captureScreenshot` CDP commands. Output must be written to a native Windows
path (`C:\Windows\Temp\...`) — a WSL-side path like `/tmp/...` passed to the Windows `node.exe`'s
`fs.writeFileSync` does **not** resolve — then `cp /mnt/c/Windows/Temp/<file> <wsl-path>` to pull it
back. This is what lets a genuinely interactive check (click a toggle, screenshot the revealed
panel) succeed, where `chrome.exe --headless --screenshot=...` alone only captures the initial page.

## `tmux` — not `expect`, not raw stdin piping — reliably drives a genuinely interactive TUI for black-box QA

Piping input (`printf '/cmd\n' | some-tui ...`) doesn't reach a TUI that needs real TTY/raw-mode
behavior — output shows only the startup banner, no command response. `tmux new-session -d ...` +
`tmux send-keys ...` + `tmux capture-pane -p` works cleanly for both slash commands and literal
`@`-containing free text, and reliably captures rendered pane content as evidence. `expect` was
unavailable in this environment; `tmux`/`script` were present.

## A CLI's "doctor"/health-check subcommand is not guaranteed read-only — verify before running one reflexively as a pure status probe

`kiro-cli doctor`, run expecting a pure `Auth ✔`-style status check, auto-remediated shell
integration by **appending** a sourcing block to `~/.bashrc` and `~/.profile` (idempotent, guarded
by an `if [[ -f ... ]]` line, but still an unannounced environment mutation outside any repo). A
command's prior framing as a status check doesn't guarantee side-effect-freedom — check what a
"doctor"/"check"/"diagnose" subcommand actually does (read its `--help`, or its source if
available) before running it reflexively during environment probing.
