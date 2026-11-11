# Shell extractor

Turns shell scripts into bundle records with `tree-sitter-bash` (extra `extract`). Plan:
`docs/plans/connected-code-graph-p1-extraction.md` P1-10; record shapes: `docs/plans/connected-code-graph-bundle-spec3.md`
(§4.1 rule 4 and "Shell", §5.4, §5.6, §5.8, §5.9, §5.10, §7).

```python
from extractors.shell import extract_shell, SourceRefused
records = extract_shell(unit_root, files, repo_id=..., unit_id=..., history=blame_fn, warnings=sink)
```

`files` are repo-relative POSIX paths that resolve under `unit_root` (the checkout or stage directory). The result is the
records of those files, with no header: `file`, `symbol`, `param`, `callsite`, `valueSet`, `constant`, `data`. The
header, its counts and the `lastChanged*` history belong to the caller's assembly (P1-44). `history(path, [(lineStart,
lineEnd)])` returns `(lastChangedAt, lastChangedCommit)`; without it symbols carry no `lastChanged*`, which
`ccg.bundle.validate` requires.

## What it extracts

| Construct | Record |
|---|---|
| the script | one `<script>` symbol (kind `script`, span = the whole file), no `parentId` |
| `name() {}`, `function name {}` | a `function` symbol `path::name`, always top level (nested definitions too); a redefinition is `name#2` |
| `$1`..`$N`, `${N}` | `param` index `N-1`, name `$N`, one per number up to the highest one the scope mentions |
| `"$@"`, `$*` | one `param` named `$@` at index N (N = the highest numbered one mentioned); its value is `spread-argument` |
| a command | `callsite`, kind `command`, `calleeName` as written (`<dynamic>` if computed) |
| a command naming a function of the same file | `resolved`, `confirmed`/`resolved-call`, `targets` = its definitions |
| any other command | `unresolved`, `possible`/`name-match` |
| `eval`, a computed command name, a computed `source`/`.` path, an argument using `${!name}` | `possible`/`shell-dynamic` |
| the leading literal of the one argument of `eval` (`eval "mcp-call $TOOL"`, `eval 'a "b c"'`) | the `eval` site as above **plus** one inner call site per complete simple command in it (below) |
| each argument `a0`.. (at most 32) | a `valueSet`: `const`, `template`, `ref`, `unknown` alternatives |
| a once-assigned script variable | a `constant` (`path::NAME`); functions read it as `{"ref": ...}` |
| a value that comes from `$N`, `$@` or a command substitution, through assignments | `data` (use = the call site `a<i>`, def = the `param` `p<k>` or the inner call site `r`), `confirmed`/`intra-flow`; `possible`/`shell-dynamic` through `${!n}` or into `eval` |

**`eval` strings** (P1-10 10b). `eval` joins its arguments and runs the result as shell, so the leading literal of an `eval`
with exactly one argument is parsed with tree-sitter-bash (never split on spaces: `eval 'a "b c"'` is `a` with the one argument
`b c`) and each complete simple command in it becomes a call site of its own: `calleeName` the command word, arguments `a0`..
(literal, or `unknown`/`runtime-built` for an expansion), `possible`/`shell-dynamic` whatever the arguments (a command naming a
function of the file resolves to it, `confirmed`/`resolved-call`, as the schema ties `resolved` to `confirmed`). `eval "mcp-call
$TOOL"` is `mcp-call` with `a0` unknown, the shape `mcp-call $TOOL` has. Numbering: inner sites come after every written site
of the scope and take the next ordinal of their callee, in source order of the `eval`s, so a written site is never renumbered
and a file with no `eval` string is numbered as before. Rules: the literal is cut at `MAX_EVAL_CHARS` (4 096) and at most
`MAX_EVAL_COMMANDS` (8) commands are taken; a word is complete only when a delimiter follows it inside the literal (an open
argument becomes one `unknown`, an open command name gives no site); a command followed by a space and then an expansion takes
one trailing `unknown` argument. No inner site for an argument that starts with an expansion (`eval "$(ssh-agent)"`,
`eval "$CMD x"`; a variable's value is never substituted), a literal that does not parse cleanly (an open quote, a trailing
`&&`), a command name that is not a plain literal, an `eval` inside the string, an `eval` with several arguments
(`eval mcp-call reindex`: the pack reads that form from the written command), or a compound command's body
(`if`/`for`/`while`). Only the `eval` site gets `data` records; an inner site has none.

**Values.** Quoting is resolved: single quotes are verbatim, `"\$1"` is the text `$1`, `a"b"c` is `abc`. An interpolated
string is a `template`. A variable resolves, within its own scope, to the assignments that reach the use (an
unconditional one kills the earlier ones; one in an `if`/`case`/loop/`&&`/pipeline/subshell, or a `+=`, is added; more than 16
reaching is `value-set-cap`). `${v:-d}`, `${v-d}`, `${v:=d}` and `${v=d}` are the alternatives *v or d*: the default is a
value alternative, **never a `bindValue`/`default`** (W3), and this extractor writes no `configEntry` or `launchFact` (those are
P1-11's). `read`, a `for` variable and `local x` define a variable with no recoverable value (`runtime-built`); a variable
nothing in the scope defines is `unresolved-ref`.

**Secrets.** A literal that fails `bundle.screen_fails`, a value assigned to a variable whose name fails the name rule
(`TOKEN=`, `DB_PASSWORD=`), a value after a flag whose name fails it (`--password x`) and a `--secret-key=x` word are withheld as
`unknown` (`runtime-built`); such a variable is never a `constant`.

## Input is untrusted

A script is parsed, never run, `source`d or `eval`-ed, and a `source`d file is not followed. Bounds (module constants):
file size `MAX_FILE_BYTES` (2 MiB), tree nesting `MAX_DEPTH` (120), arguments per command 32, positional index 64,
records per file 100 000, one literal 8 192 characters, an `eval` string 4 096 characters and 8 commands, an interpolated string 64 parts, 16 alternatives, variable chains 8 deep,
parse time 20 s. Walks use an explicit stack.

| Condition | Outcome |
|---|---|
| name is empty, absolute, not normalized, a symlink out of `unit_root`, not a regular file; over `MAX_FILE_BYTES`; not UTF-8 | `SourceRefused`: the unit fails, nothing is skipped (index D12) |
| tree-sitter reports an error/missing node; nesting over `MAX_DEPTH`; parse timeout | warning (`parse-error`, `too-deep`, `parse-timeout`), the file keeps its `file` and `<script>` symbol and has no calls |
| record cap reached | warning `record-cap`; the remaining commands of the file are dropped |

Warnings go to the `warnings` list as `{path, code, line, detail}`; the shell `skippedFiles` enum does not apply.

## Not modelled (limits)

- **One tree-sitter ERROR discards every function and call in the file** (architect-accepted limit): the file keeps its `file` and
  `<script>` symbol and one `parse-error` warning. On `git ls-files '*.sh'` that is 5 of 113 files (about 4.4%):
  `claude/teco/hooks/test-ensure-environment.sh`, `cpg/eval/headless/tests/test_launcher.sh`, `cpg/sandbox/offline-run.sh`,
  `cpg/sandbox/tests/test_sandbox.sh`, `skills/joern-cpg/scripts/test-stamp-wiring.sh`; the causes are `(( 0$(...) ))`,
  `<>/dev/tcp` and `<<<`. An error-tolerant extractor is a P2 backlog item.

- No `cond`/`ret` records, so `complexity` is 1 (spec4 §2: 1 + own `cond` records).
- `shift`, `set --`, `getopts`, `$#`, arrays, `declare -n`, `export -f`, functions defined by `eval` and `bash -c` strings (an `eval` string's commands are call sites only).
- Positional parameters are per scope; argument positions are the written ones (a word that expands to several, such as `"$@"`,
  does not shift the later ones).
- A function's script-level variable is read only when it is a once-assigned `constant`; other script variables are `unresolved-ref`.
- The tree-sitter grammar differs from bash in edge cases (`$10` is read as `$1` then `0`, as bash does; brace expansion is
  not expanded).
- Command resolution is by name within the file; cross-file resolution (`source`d libraries, `PATH` tools) is the assembler's `CALLS_NAME`.
