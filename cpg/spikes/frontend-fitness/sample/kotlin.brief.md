**You are labelling call sites in a Kotlin codebase, blind.** For each call site in
`/home/mauricio/prg/graphmind-ai-lab/cpg/spikes/frontend-fitness/sample/kotlin.jsonl` you write down which functions or methods it can really call, by reading the
source. Your answers are compared with tools later; you will not see their output, and you must
not use any tool that resolves code for you.

**What you have.** The codebase at `/home/mauricio/prg/graphmind-ai-lab/cpg/.ccg-work/corpora/kotlin/src` (read-only), the sample file, and your output
file `/home/mauricio/prg/graphmind-ai-lab/cpg/spikes/frontend-fitness/labels/kotlin.jsonl`. Use only file reading, `grep`/`rg` and `find`.

**Which call a sample line means.** Each line gives `path`, `line`, `column`, `callText` and a short
excerpt. The site is the call whose **callee expression** (the part before the argument list, such
as `self.repo.get` in `self.repo.get(x)`) starts at `line:column` and reads exactly `callText`.
In `a.b().c()` two sites start at the same place: `callText` `a.b` is the inner call, and `a.b().c`
the outer one.

**What you must not do.** Do not run any program that understands code: no Joern or `*2cpg`, no
indexer, no language server (Pyright, `tsc`, `gopls`, clangd, Roslyn, OmniSharp), no compiler or
build tool, no `go doc`/`go vet`. Do not use the web, MCP tools or any knowledge base. Do not open
anything outside `/home/mauricio/prg/graphmind-ai-lab/cpg/.ccg-work/corpora/kotlin/src` except the sample file and your output file. Using any of these
voids your work.

**The corpus boundary.**
- **In the corpus** (a call into these is *internal*): every file under `/home/mauricio/prg/graphmind-ai-lab/cpg/.ccg-work/corpora/kotlin/src`, tests
  included, generated files included (`KtorAdminLibrary/src/main/kotlin/ir/amirroid/ktoradmin/utils/FileUtils.kt`).
- **Vendored code** is someone else's library kept in the repository: any file with a directory
  named one of `vendor`, `vendored`, `_vendor`, `third_party`, `third-party`, `node_modules` anywhere in its path. A call into it is *external* (the package
  form below).
- Everything else, the language's standard library and installed packages, is *external*.

**What to write for each site: the true callees.**
- A **direct call** to a function or method with a body: that one definition.
- A **method call on an object**: work out the object's declared type by reading (its parameter
  or field type, or the assignment that created it). List the method that type has or inherits
  from its nearest in-corpus ancestor, **plus every in-corpus override or implementation** in a
  subtype of that type (grep for subclasses and implementors). If the type's method has no body
  (an interface, abstract or protocol method, a pure virtual, a TS `type`-alias method signature),
  list it as `declared` and list the
  implementations as `internal`. If the method is **inherited from a base class outside the
  corpus** and the type does not override it, list it as `external` (that base's package), plus any
  in-corpus overrides in subtypes as `internal`. If you cannot work out the type, mark the site
  `unsure` and list what you found.
- **Special declarations.**
  - A primary constructor (Kotlin, C#), or a Java record header when the record has no explicit
    canonical constructor: `declared` `Type.<init>`, `defLine` = the header line.
  - A C++ `= default` member: `declared`.
  - A Kotlin `expect` declaration: label its `actual` as `internal` instead.
  - A C# `partial` method: label the implementing declaration.
  - A C/C++ prototype, or an in-class declaration of a member defined out of line: label the
    definition.
  - A Go call through an interface value: `declared` at the interface that declares the method
    (`IfaceType.NAME`; for a method that comes from an embedded interface, that interface's own
    spec), plus every in-corpus implementation as `internal`. For an unnamed interface literal
    (`interface{ Close() error }` in a signature), still list the spec as `declared`.
- A **call through a variable holding a function** (a callback, a parameter, a stored handler):
  trace where the value comes from, up to two levels of in-corpus callers, and list every
  in-corpus definition that can arrive there. If it goes further, mark `unsure` with the partial
  list.
- **Constructors, macros and JSX:** see the table. Decorator and annotation applications (`@x`,
  `@x(…)`, `[Attr]`) are never meant to be in your sample. If a line points at one, mark it
  `notACall`.
- If **no target at all** can be determined from source (a name computed at run time, `eval`,
  reflection on a computed string), set `dynamic: true`. If some targets can be determined, list
  them too.
- If the site is **not a call** (a type conversion, a macro that expands to no call, a
  lower-case JSX tag), set `notACall: true`.

**Writing a target.**
- `internal`: `path` (relative to `/home/mauricio/prg/graphmind-ai-lab/cpg/.ccg-work/corpora/kotlin/src`, `/` separators), `qname` (the grammar in the
  appendix: `path::Scope.name`), and `defLine`. `defLine` is the 1-based line of the
  definition's **name** (the line with `def name`, `function name`, the method name, the class name
  for a type); decorators and annotations above it do not count. For an anonymous function it is
  the line where the function starts. For a Kotlin primary constructor it is the class header
  line. A function whose name exists only after macro expansion gets `"qname": null`, with the
  path and line where the macro is used to define it.
- `declared`: the same three fields, for a bodiless static target.
- `external`: `package`, in the form the table gives.

| Language | `qname` form (appendix) | External `package` | Constructor target | Other kinds | `notACall` |
|---|---|---|---|---|---|
| Python | `path::Class.method`, `path::func`, nested `path::outer.inner` | the top-level module as imported (`fastapi`); standard library `stdlib:<module>` (`stdlib:json`); builtins and methods of built-in types (`len`, `dict.get`, `str.join`) `builtin` | `Foo(…)` → `Foo.__init__` in `Foo` or its nearest in-corpus base; if none, the class `Foo` itself | — | none expected |
| TS/JS | `path::func`, `path::Class.method`, `path::Class.<init>`, binding-named arrows `path::outer.name` | the npm specifier, scope kept (`react`, `@tanstack/react-query`); Node core `stdlib:<module>`; JS and DOM globals and built-in type methods (`fetch`, `JSON.parse`, `Array.prototype.map`) `builtin` | `new Foo(…)` → `Foo.<init>` in `Foo` or its nearest in-corpus base; if none, the class `Foo` | `<Foo …/>`: a function component's function; a class component's `Foo.render`; `<A.B/>`: member `B`; `memo(X)`/HOC-wrapped: trace to `X` | lower-case (intrinsic) JSX tags |
| Java | `path::Class.method`, `path::Outer.Inner.method`, `path::Class.<init>` | the package of the imported type (`org.springframework.web.bind.annotation`); `java.*`/`javax.*`/`jdk.*` `stdlib:<package>` | `new Foo(…)` → the `Foo.<init>` overload that matches (by `defLine`); with no explicit constructor, the class `Foo`. `super(…)`/`this(…)`: the matching `<init>` | annotations are not call sites | none expected |
| Kotlin | `path::func`, `path::Class.method`, `path::Class.Companion.f`, `path::Class.<init>`, `path::Class.prop.<get>` | as Java; `kotlin.*` `stdlib:<package>`; scope functions (`let`, `apply`, `also`) `stdlib:kotlin` | `Foo(…)` → the matching `Foo.<init>` (primary: class header line); none explicit: the class | trailing-lambda DSL calls: the function called, not the lambda | none expected |
| C# | `path::Ns.Sub.Class.Method`, generic types ``Box`1``, `path::Ns.Class.<init>` | the type's namespace (`Microsoft.AspNetCore.Mvc`); `System.*` `stdlib:<namespace>` | `new Foo(…)` → the matching `Foo.<init>`; none explicit: the class | attributes are not call sites | none expected |
| Go | `path::Func`, `path::Type.Method` (receiver base type, no `*`) | the import path (`github.com/spf13/cobra`); standard library `stdlib:<path>` (`stdlib:net/http`); `len`, `append`, `make`… `builtin`. Vendored `vendor/<module>/…` → that module path | Go has none: `NewFoo()` is a plain function; composite literals are not call sites | call through an interface value: every in-corpus type's method that implements it, by method name and signature | type conversions `T(x)` |
| C/C++ | `path::Ns.Class.method`, out-of-line `path::Ns.Class.method`, C `path::func`, `path::Ns.Class.<init>` | the header the declaration comes from: standard `stdlib:<header>` (`stdlib:vector`, `stdlib:stdio.h`); other out-of-corpus headers `header:<include text>` | `new Foo(…)` or `Foo(…)` → the matching `Foo.<init>`; none explicit: the class | function-like macro `M(…)`: the in-corpus functions its expansion calls directly, expanding nested macros, across every `#if` branch; only external calls → those externals | a macro whose expansion calls nothing |

**Output.** One JSON object per line in `/home/mauricio/prg/graphmind-ai-lab/cpg/spikes/frontend-fitness/labels/kotlin.jsonl`, in **ordinal order, never skipping a
site**, appended as you go:

```json
{"ordinal": 17, "siteId": "<as in the sample>",
 "callees": [{"kind": "internal", "path": "src/a/b.py", "qname": "src/a/b.py::Repo.get", "defLine": 42},
             {"kind": "external", "package": "falkordb"}],
 "notACall": false, "dynamic": false, "confidence": "sure",
 "evidence": "receiver self.repo: Repo (src/a/svc.py:12); Repo.get b.py:42; no subclass of Repo (grep)"}
```

- `kind` is `internal`, `declared` or `external`. `confidence` is `sure` or `unsure`. **`unsure` is
  always better than a guess or a skip.**
- `evidence`: at most 300 characters, citing file:line for what you relied on.
- **Stop** after the line that brings the number of lines with at least one `internal` or
  `declared` callee to
  **150**, or after ordinal **500**, whichever comes first.
- **Resume:** if you are restarted, read the last ordinal in `/home/mauricio/prg/graphmind-ai-lab/cpg/spikes/frontend-fitness/labels/kotlin.jsonl` and continue with the
  next one. Never rewrite earlier lines.

**Appendix: `qname` grammar** (bundle spec §4.1 and §4.2 Kotlin, version 1.2 (`docs/plans/connected-code-graph-bundle-spec3.md`; `-spec4` amends nothing in §4), inserted
verbatim by S25).

### 4.1 Common grammar

```
qname    = path "::" chain
path     = the defining file's repo-relative POSIX path (§8.2)
chain    = segment *( "." segment )
segment  = base [ dup ]
base     = NAME | "<anon>" "#" N | "<init>" | "<module>" | "<script>" | "<document>"
         | NAME "." ( "<get>" | "<set>" | "<del>" )        ; accessors: two segments, the property then the accessor
dup      = "#" N                                           ; N ≥ 2, only on a NAME base
N        = decimal integer, no leading zero
NAME     = the identifier exactly as written in source (case kept, NFC), with any quoting
           removed (Kotlin backticks, C# @verbatim)
```

**Rules that hold in every language:**

1. **The path is part of the `qname`.** Rejected alternative: a language-native dotted name such
   as `falkorchat.storefront_api.build_storefront_router.health`. It cannot be written from
   source alone: it needs the import root, `go.mod`'s module path or tsconfig `paths`. It also
   collides across units of one repo (two `tests/conftest.py` both give `tests.conftest`), and the
   symbol id has no `unitId` to separate them (F2). With the path, a moved build root leaves ids
   unchanged and a moved file changes them, which is the move rule proposal §3.8 records.
2. **Scopes.** A segment is added for each enclosing named scope that **contains** the
   definition: types (class, interface, struct, enum, record, trait, object), functions and
   methods, and the namespace-like scopes listed per language. `if`/`try`/`with`/loop blocks,
   comprehensions and plain blocks never add a segment.
   - **Containment is by evaluation scope, not by text** (analyst F2(b)). Decorator, annotation
     and attribute expressions belong to the scope **enclosing** the decorated definition, even
     though its span includes them (§5.3). So does every type annotation.
   - **Parameter default values** belong to the enclosing scope in Python, where they are
     evaluated at definition time. In JS/TS, Kotlin, C# and C++, where they are evaluated at call
     time, they belong to the function itself.
   - Example: in `@pytest.mark.parametrize(…, ids=lambda c: …)` above `def test_x`
     (`falkor-chat/server/tests/test_fusion.py:241`), the lambda is a module-level `<anon>#N`,
     counted with the module's other anonymous scopes, and not `test_x.<anon>#1`.
3. **Top level.** The chain begins with the first named scope. A top-level function `f` is
   `path::f`, never `path::<module>.f`.
4. **One synthetic file symbol.** Every code file has exactly one file-scope symbol: `<module>`
   (code), `<script>` (shell) or `<document>` (Markdown). Its span is the whole file. Code outside
   every function and type belongs to it. Code in a type body but outside any method (field
   initializers, Python class-body statements, Java static or instance blocks, Kotlin `init {}`)
   belongs to the **type's** symbol, except as below.
   - **Exception: a declared primary constructor owns the initializers** (analyst N4). In a
     Kotlin class whose header declares a primary constructor, and in a C# class, struct or
     record with a primary constructor, the `init {}` blocks (Kotlin) and the **instance** property
     and field initializers (both) belong to the primary constructor's `<init>` symbol, not to the type.
     That is what both languages execute: those regions run as the primary constructor's body,
     and its parameters are in scope there. Its `lineStart`/`lineEnd` stay the header's parameter
     list; the initializer regions are its **owned regions** (§5.3 `ownedRegions`, §3.5 `line`).
     Without the exception, `class Orders(r: Repo) { init { r.load() } }` has the `r` param on
     `Orders.<init>` and its use on `Orders`, and §5.10's same-symbol rule cannot express the
     flow. A Kotlin class with no declared primary constructor keeps rule 4 as stated, and so do
     C# `static` field and property initializers, which belong to the type (analyst P3-2). A C#
     primary-constructor parameter used inside a **method** is a capture, a cross-symbol fact
     `data` does not express (a known limit, like any field read).
5. **Anonymous scopes are `<anon>#N`.** These are lambdas, function literals, anonymous classes
   and object expressions. `N` counts from 1, in source order (by start line, then start column),
   over the anonymous scopes with the **same parent chain**. Nesting is ordinary:
   `f.<anon>#1.<anon>#2`.
6. **Constructors are `<init>`**, in every language that has them. Python's `__init__` is
   **not** renamed: it is an ordinary method as written.
7. **Duplicates get `#N`.** When two or more symbols have the same parent chain, the same base and
   the same signature (or none), the first in source order keeps the bare base, and the k-th gets
   `#k` (k ≥ 2). Examples are a redefinition, a conditional definition in both branches, a static
   and an instance member with the same name in JS, and Go's several `init`s.
8. **Which definitions are symbols** (analyst F2(a), F4; S03 Pass 2 N1, N5). Two kinds are
   symbols, and nothing else is:
   - every function-like definition with a body;
   - every **bodyless declaration a call can resolve to** that is named in the "symbols" column
     below. Frontends resolve calls to such declarations (Joern resolves `embedder.embed(…)` to the
     `Protocol` stub); if they were not symbols, `callsite.targets[]` would name ids with no record.

   **The table is exhaustive:** a bodyless declaration form it does not name is **not** a
   symbol. Each row is decidable from the declaring file alone (§4.0). The organizing reason for
   the right-hand column is that each entry is either a **redeclaration** of an entity whose
   executable declaration is elsewhere (that declaration is the symbol), or **field-like**, or
   reaches no well-formed call.

   | Language | Bodyless declarations that **are** symbols | Bodyless declarations that are **not** |
   |---|---|---|
   | Python | methods of a `Protocol` class and `@abstractmethod` methods, `...`-bodied stubs included (`falkor-chat/server/falkorchat/embedding.py:37`) | `@overload`/`@typing.overload` stubs; all `.pyi` content |
   | TS/JS | `abstract` methods; method signatures (`m(): T`) in an `interface` body, or in an object type literal that is the whole right-hand side of a `type` alias (`T.m`) | overload signatures; `declare` (ambient) declarations; all `.d.ts` content; property signatures of function type (`f: () => void`, field-like); call, construct and index signatures (no name) |
   | Java | `abstract` methods; bodyless interface methods, annotation-type elements included; `native` methods; a record's header, as its canonical constructor `<init>`, when the record declares no explicit canonical or compact constructor (span: the header's component list); each record component with no explicitly declared accessor, as its implicit accessor method `R.x` with signature `()` (span: that component's header entry; analyst P3-3), so an explicit accessor added later keeps the same id | — |
   | Kotlin | abstract and bodyless interface functions; `external` functions; the primary constructor `<init>`, when the class header declares one (`class A(…)`, `class A()` or `class A constructor(…)`; span: the header's parameter list) | `expect` declarations (the `actual` is the symbol); default accessors (field-like) |
   | C# | abstract and bodyless interface methods; `extern` methods; the primary constructor `<init>` of a class, struct or positional record (span: the type's parameter list) | a `partial` method's bodyless defining declaration (the implementing declaration is the symbol); auto-accessors and other bodyless accessors (`{ get; set; }`, abstract or interface properties), field-like |
   | Go | **interface method specs** of a named interface type (a `type` declaration, defined or alias), as `IfaceType.NAME` (§4.2 Go); bodyless function declarations (assembly-implemented) | method specs in an unnamed interface type literal (`func f(c interface{ Close() error })`): no source name to key on; type-set terms of constraint interfaces |
   | C/C++ | pure-virtual in-class declarations (`= 0`), whether or not an out-of-line body also exists; `= default` declarations, in-class or out of line | `= delete` declarations (no well-formed call resolves to one); every other bodyless in-class member declaration (its out-of-line definition, wherever it is, is the symbol); free-function prototypes; `extern` declarations; bodyless `friend` declarations; function-like macros |

   - **C++ is decided from the declaring file alone** (analyst N1). Whether `void set(int) const;`
     in `motor.h` is a symbol no longer depends on whether some `.cpp` defines it: it is not. A
     pure virtual with an out-of-line body gives **two** ids, at two paths
     (`motor.h::hw.Motor.f` and `motor.cpp::hw.Motor.f`).
   - **Resolution into a non-symbol.** `targets[]` (§5.6) carries only ids of symbols under this
     rule. When the frontend resolves a call to a non-symbol declaration, the normalizer
     substitutes the entity's symbol if the unit's CPG holds it (a C/C++ prototype or in-class
     declaration → its definition; a C# partial defining declaration → its implementing one), and
     otherwise drops the target. The same applies to an implicit or compiler-synthesized member
     (an undeclared default constructor). If `targets` ends up empty, §5.6's `possible`/`name-match`
     applies, and `resolution` **stays `resolved`** (analyst P3-n2): it records what the
     frontend did, so `resolved` with empty `targets` is exactly "resolved to a non-symbol". **[P-later]** for the non-slice languages; how Joern's frontends model these is not
     verified here.
   - **Vendored code is never a symbol** (§5.0, "Vendored paths").

   Types are symbols even without code, so field contract points have a type `qname` to key on
   (§6).
9. **A segment need not be defined in the same file.** Examples are a Go method on a type declared
   in a sibling file, and a C++ out-of-line member definition. The path is the **definition's**
   file.
10. **`name`** (in the `symbol` record) is the last segment's base, without `#N` or an accessor
    suffix. For `<init>`, `name` is the type's name. `nameNorm` is defined in §1.2.

#### Kotlin — **[P-later]**

- **Scopes:** `class`, `interface`, `object`, `enum class`, `companion object` (named
  `Companion` unless it is given a name), functions, and local functions.
- **Top-level functions** sit directly under the path.
- **Special names:** the primary and secondary constructors are `<init>`. A declared primary
  one is a symbol even with no body (§4.1 rule 8), and it **owns** the class's `init {}` blocks
  and property initializers (rule 4's exception). Without a declared primary constructor,
  they belong to the type symbol. Property accessors **with a body** are `prop.<get>`/`prop.<set>`. Lambdas and `object :` expressions are `<anon>#N`.
- **Extension functions** use the plain function name; the receiver goes into the signature
  (§4.3). Backticked names are written without the backticks.

```kotlin
// fx-svc-kt/src/main/kotlin/acme/Orders.kt
fun String.slug(): String = …            // Orders.kt::slug                  sig  String.()
class Orders(val repo: Repo) {           // Orders.kt::Orders  (type);  primary ctor  Orders.<init>  (Repo)
  init { repo.warm() }                   // owned by Orders.<init>: a call site of it, not of the type
  companion object { fun of(): Orders }  // Orders.kt::Orders.Companion.of   sig  ()
  val total: Int get() = …               // Orders.kt::Orders.total.<get>    sig  ()
  fun place(o: Order) { list.forEach { … } }   // Orders.place (Order)  and  Orders.place.<anon>#1 (no signature)
}
```
