# Testing a guard — techniques for static-analysis tripwires

> **On-demand knowledge base for `tdd-engineer`.** How to test a *guard* — an AST reader, lint
> rule, grep check or wiring assertion whose subject is other code's **text** rather than its
> behaviour. These are the shapes that pass their own tests while missing the thing they name.
> Consult before delivering or hardening one; ordinary unit/integration practice lives in
> `tdd-engineer.md`.
>
> Origin: distilled 2026-09-09 from `kaizen_team` via `agent-maintenance` skill §5.

## A mutation test and a coverage probe answer different questions — only the probe finds the recurring defect

A **mutation test** asks *"does my reproduction die?"* — you write the bad thing you have in
mind, run the guard, require red. A **coverage probe** asks the opposite question: *"what can
this reader **not** see?"* — you enumerate every syntactic form of the thing the guard claims to
catch, run the **delivered** reader over each one, and list the misses.

Mutation testing is necessary and is not sufficient, because its enumeration is your imagination.
The evidence is stark: across a chain of 13 instances of one guard defect, five were found only
by full review cycles — and every one of those had been shipped by a conscientious,
mutation-testing implementer. The first instance caught *before* a gate was caught by a probe,
inside the unit that created it, for the price of one script.

**Two design rules make a probe age correctly.**

1. **Derive the enumeration from the language/runtime, not from a list you wrote down.** A
   hand-written list silently opens a hole when the language grows; a derived one **reddens**.
   For a Python binding-form reader that means walking `ast`'s own class tree rather than naming
   node types.
2. **Ship the probe as a test, not as a one-off script.** A probe that lives in a scratch file
   certifies one revision and then stops existing.

**The derivation has a parameter, and it must be stated.** Intersecting every `ast.AST`
subclass's `_fields` against a set of binding-field names is the right method, but the answer
depends on which names you declare. Re-derived 2026-09-09 on CPython 3.12.3 (system `python3`),
two definitions of the same census:

| field set | classes |
|---|---|
| `target, targets, optional_vars, name, names, asname, arg, rest` | **27** |
| `target, targets, optional_vars, name, arg` | **22** |

The five-name set drops `Global`, `Import`, `ImportFrom`, `MatchMapping`, `Nonlocal`. So "the
grammar has N binding forms" is not a fact on its own — cite the field set beside the number, or
the next reader cannot tell a hole from a different question.

**And a mutant whose kill signal collides with a harness failure proves nothing.** Bash exits
**127** when `set -u` aborts on an unbound variable — the *same* status as command-not-found.
Re-derived 2026-09-09, bash 5.2.21(1) on WSL2: `set -u; echo $NOPE` at top level, the same inside
a function, and the same as `for k in $NOPE` all exit **127**, identical to `set -u; nosuchcmd_xyz`
and to calling a function whose definition was deleted. So on a shell harness the **exact-rc**
oracle — which is the right repair for a `rc != 0` oracle, and is assumed elsewhere in this file —
still cannot separate *"the guard is gone"* from *"the helper is undefined"*, and a mutant built by
`unset`-ing the variable additionally proves nothing about a guard that only **reads** it. Use the
**set-but-empty** shape: it kills honestly at rc **0**, with the guard's own output emitted
(`NOT k IN []`), so the red is attributable to the guard rather than to the harness. The portable
rule: pick a mutant whose failure signal is **distinguishable from every way the harness itself can
fail**, and establish that before reading a red as a kill — otherwise the mutation test degenerates
into the same unfalsifiable green the exact-rc oracle was introduced to fix.

### The worked example — a five-generation arc, closed by a probe

`skills/joern-cpg/scripts/test-stamp-wiring.sh`'s `rq` call-site check is the reference
implementation of everything above, and it earned that by being the *fourth* generation of the
defect it now closes. Each of the first three was fixed by widening the body and leaving the
sentence alone; generation four passed four named **mutation** shapes and still missed a literal
`GRAPH.DELETE` written across a backslash continuation — its reader was line-oriented, and a
shell call site is not. Closed 2026-09-09 (`4df5e45`) by `graph-dba`. Four things make it worth
copying:

1. **It took *both* closures, each where it belonged.** It **widened** where the miss was a wrong
   *unit of analysis* — joining continuations, anchoring on `rq[[:space:]]` — which closed four
   forms in nine lines. And it **narrowed the claim** to exactly what remains: six forms stay
   blind (wrapper forwarding `"$@"`, `GRAPH".DELETE"`, `GRAPH\.DELETE`, `"$CMD"`,
   `"${CMDS[0]}"`, and a command not spelled `GRAPH.…` at all — `rq "$Q" PING`), written as a
   bound with a check-it-by-hand instruction rather than implied.
2. **The mechanism is a coverage probe, not a mutation suite** — 36 forms on three axes
   (invocation syntax; how the command argument is spelled; text that only *looks* like a call
   site), and each row is adjudicated **twice**: bash says whether `rq` really receives a
   non-query command, the delivered reader says whether it flags it. A mis-written row therefore
   fails the probe instead of silently certifying a form that was never a call site. The reader
   is extracted into one function the probe calls, so the probe cannot certify a re-typed copy.
3. **The claim is pinned by the mechanism.** Widening the reader without rewriting the stated
   bound turns a `blind` row **red**. That is the structural answer to a guard whose stated reach
   exceeds its mechanism: make the sentence a thing the suite can falsify, and the gap cannot
   reopen silently.
4. **The probe caught generation five in flight, inside the fix for generation four.** The first
   continuation-joiner joined with a space; bash joins with **nothing**, so `rq "$Q" GRAPH\` +
   `.DELETE` is one word to bash (executed) and reads as `GRAPH .DELETE` to the reader (no token,
   silent pass). The probe caught it *while its author was writing the claim*, and form A15 now
   pins it. **The portable rule, not just the anecdote:** any reader that reconstructs logical
   lines from a shell source must join continuations with *nothing* — a space manufactures a
   token boundary bash never creates, and it does so precisely at the place the reader is being
   hardened.

**Verified here by execution, 2026-09-09**, against the delivered reader extracted verbatim: the
continuation form that previously scored `PASS all 3` is now flagged (`GRAPH.DELETE@451`) with the
clean tree still at `SITES=3` and no tokens; three of the six stated-blind forms reproduce as
blind with bash confirming `rq` really receives `GRAPH.DELETE`; and two widenings (tab-separated
call, single-quoted literal) flag correctly.

**And the stopping rule that makes narrowing legitimate — this is the transferable half.** You may
stop widening when **a miss is bounded to a false failure rather than a false pass.** Here it is:
`rq` fails closed twice over on a non-query command, because it appends the cypher as a third
argument and then requires a statistics trailer on the reply's last line. Measured, with a passing
control: `PING` → rc 1 (`ERR wrong number of arguments`), `INFO` → rc 1 (empty reply),
`GRAPH.RO_QUERY` → rc 0 with `Query internal execution time: …`. The static check is therefore a
**lint over a property the runtime already enforces**, not the safety boundary itself — which is
precisely what licenses it to stop at a stated bound instead of chasing a shell parser. Establish
that cost asymmetry before you accept a narrow claim; without it, narrowing is just conceding.

**One retraction — right about one file, wrong about the other.** Probing that bound, I found
`PING`, `INFO` and `keys` — each one contiguous literal at the call site, each genuinely passed to
`rq` by bash, none flagged — drafted it as generation six, then withdrew it: the mechanism is
scoped to runs matching `/GRAPH\.[A-Za-z_.]*/`, and the bound's first clause reads *"a command that
is not one contiguous `GRAPH.<word>` run of characters at the call site"*, which covers them
literally. That is true of `test-stamp-wiring.sh` — the file I read. It was **false of
`pipeline.sh`**, whose copy of the same sentence said only *"not one contiguous literal"*, the
`GRAPH.<word>` scoping dropped, putting `rq "$Q" PING` inside its stated reach and outside its
mechanism. So the finding I withdrew was a real defect, in the file I had not opened; `graph-dba`
closed it 2026-09-09 (`00bebdc`) by converging both files on the scoped claim, tightening a
*second* sentence in `test-stamp-wiring.sh` that had drifted the same way, and pinning the scoping
with probe form `B15` (`rq "$Q" PING`, `blind`). Two rules, and the second is the stronger:
**read the stated bound before filing against it** — a finding drawn from the examples rather than
the claim is a finding about the documentation's illustrations; and **checking a bound means
checking every place it is stated**, because one bound written in two files is one claim with two
chances to go stale.

## A hand-written "which object is this" resolver has two axes, and only one of them is finishable

Any guard that resolves *what a name refers to* is answering two separate questions, and
hardening one while stating the other semantically regenerates the same defect forever:

- **TARGET axis — which binding forms bind a name** (`Assign`, `AnnAssign`, `NamedExpr`, `For`,
  comprehensions, `withitem`, `MatchAs`, …). This axis is **finishable**: derive it from the
  runtime as above and enumerate it completely.
- **VALUE axis — which expressions denote the object.** This is alias/points-to analysis and a
  hand-written reader **cannot** close it. Each closure spawns the next: `x = self`, then tuple
  unpacking, then a conditional expression, then a container round-trip, then a call argument.

The practical consequence is a trap: **a coverage probe that varies only the target axis while
holding the value axis at one spelling comes back empty and certifies nothing.** That is a
green result produced by not asking the question.

Worked instance — a router-reach guard over a service layer, with the injection placed on a
method the router reaches:

| injected call site | suite |
|---|---|
| `svc = self._services` then `svc.start_workflow_run(…)` | **1 failed / 184** |
| `me = self` then `me._services.start_workflow_run(…)` | **185 passed** |

The reader sees an alias of the service **attribute** and is blind to an alias of the
**receiver** — one hop up the same expression, and outside any enumeration of binding forms.

**The rule when the value axis cannot be closed: stop widening.** State the rule narrowly and
truly — name the exact resolution mechanism the code performs (*"`ast.unparse(value)` matches one
of these literal prefixes"*), not the semantic reach you wish it had — and move the load-bearing
evidence to a **behavioural** test. A tripwire needs a rule that is narrow and true, never one
that is broad and complete. Measure and record the residual (*"zero receiver-alias bindings
across the package today"*) so the narrow claim is a decision, not an omission.

**The opposite polarity, on the same reader: where the value axis under-approximates, the
*frontier* over-approximates.** A reach guard that expands its frontier by collecting
`ast.Attribute` nodes under a prefix collects every `self.<name>` — not only the ones in call
position. So a method handed somewhere as a **value** — `executor.submit(self._run_turn, …)`, a
callback registered, a method stored on an attribute — enters the frontier exactly as a called one
does, and its body is then walked for the guarded prefix. Verified 2026-09-09 on CPython 3.12.3:
`{c.attr for c in ast.walk(tree) if isinstance(c, ast.Attribute) and ast.unparse(c.value) in
{'self'}}` over `self._ex.submit(self._run_turn, 1)` returns `_ex` **and** `_run_turn`. Two
consequences, neither of them predictable from the word *reach*: work deferred to a thread pool is
**inside** the guarded reach — you cannot escape the tripwire by moving the call into a worker
method — and a method only ever *referenced*, never called, is walked anyway. For a tripwire that
is the right failure direction, but it belongs in the docstring: a reader who takes "reach" to mean
the call graph mispredicts this reader in **both** directions at once.

## When the docstring states SEMANTIC reach and the body does a SYNTACTIC match, the gap regenerates the defect after every fix

This is the signature failure of hand-written guards. The docstring says *"every method a route
can reach, by any path"*; the body matches three hardcoded prefix strings. Every fix closes the
instance in front of it and leaves the sentence alone, so the next review pass finds the defect
**inside the artifact that closed the previous one** — three consecutive passes did exactly that
in one chain.

**The productive question is not *"does the guard fire on my reproduction?"*** — it will; you
wrote the reproduction to match the body. It is:

> **What is the smallest edit to production code that satisfies the docstring and survives the
> body?**

Ask it as one probe, before delivering. The answer is usually a local alias assignment
(`svc = self._services`) or a call one file over through an ordinary delegation — not anything
exotic. Two verified instances from the same guard: the alias survived at 183 passed, and a bare
`HTTPException` raised in a collaborator module survived at 183 passed while the guard's own
exemption string named *every route*.

**An over-claiming docstring admits two correct closures, and the defect is only ever the gap:**

1. **Extend the mechanism to the claim** — make the reader actually do what the sentence says.
2. **Narrow the claim to the mechanism** — rewrite the sentence to describe what the body does,
   and state the bound explicitly.

Both are legitimate. **Say which one you chose and why, in the artifact.** A team that has widened
several times running will not spontaneously consider retreating the claim, so if you are handed a
finding without that choice attached, make it yourself and report it rather than reflexively
widening again. Where you narrow, the bound is the load-bearing half of the comment — write it as
*what this does not cover*, not as an implication.

**Related, for the exemption strings a guard carries:** an allowlist keyed on **names** rather
than **sites** is blind to a second use of a name already in the list — so admitting one name to
let a change through silently retires the guard for every later use of that name in that file,
with no test edit needed. Site-qualify (`(enclosing_function, name)` pairs) instead.

**And write the exemption as an equality, not a subtraction.** `residual - allowed == ∅` (or an
`issubset`) admits every *further* member silently — that is the shape that lets an allowlist grow
without anyone deciding to grow it. `residual == {allowed}` makes the next unlisted member a
stop-and-decide, and it reddens in the other direction too, on an exemption left behind by a raise
that is gone. Read whole 2026-09-09 in the reference implementation
(`falkor-chat/server/tests/test_storefront_api.py`, `test_the_raises_a_route_can_reach_…`):
`set(STOREFRONT_RAISES_TODAY) - storefront_family == frozenset({"RuntimeError"})`, with that one
name pinned further by a **list** of `(function, class)` sites (`== ["enqueue_turn"]`) — a list
rather than a set, so a second raise of it inside the *same* function cannot collapse onto the
first either.
