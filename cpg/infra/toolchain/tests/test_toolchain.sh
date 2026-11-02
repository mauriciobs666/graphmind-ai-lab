#!/usr/bin/env bash
# test_toolchain.sh — regression checks for the ccg-toolchain / ccg-toolchain-truth images
# (C-1001 P0, S24). Needs Docker and both images already built (build.sh, build.sh --truth).
# Runs everything with `--network none`, in throwaway `docker run --rm` containers of the two
# images only. Exit status = number of failed checks.
#
#   cpg/infra/toolchain/tests/test_toolchain.sh
#
# Env: CCG_CONTROL_IMAGE  image of the pre-S24 `ccg-versions` (default ccg-toolchain:d3184effb566);
#                         the p2-m1 positive control is skipped if it is not present.
#
# Covers: (1) the tag is a deterministic function of the build context; (2) `ccg-versions`
# exits non-zero for the p2-m1 drop-in pair and for a `set -e` drop-in, and the S02 image is
# the control that exited 0; (3) the sandbox image carries no truth tool; (4) ccg-truth-smoke
# passes on the real smoke projects and FAILS on a mutant of each assertion (override edge
# removed; call through the interface removed), per language, plus a header-resolution mutant
# for C/C++ (the compiler's own system headers hidden).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TC_DIR="$(cd "$HERE/.." && pwd)"
BUILD="$TC_DIR/build.sh"
TC="$("$BUILD" --print-tag)"; TR="$("$BUILD" --truth --print-tag)"
CONTROL="${CCG_CONTROL_IMAGE:-ccg-toolchain:d3184effb566}"
SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/ccg-toolchain-test.XXXXXX")"
trap 'rm -rf "$SCRATCH"' EXIT

pass=0; fail=0
ok()  { pass=$((pass + 1)); printf 'ok   %s\n' "$1"; }
bad() { fail=$((fail + 1)); printf 'FAIL %s\n' "$1"; }
expect() {  # expect <description> <want: 0|nonzero> <cmd...>
  local d="$1" want="$2" rc; shift 2
  "$@" >"$SCRATCH/last.out" 2>&1; rc=$?
  if { [ "$want" = 0 ] && [ "$rc" -eq 0 ]; } || { [ "$want" = nonzero ] && [ "$rc" -ne 0 ]; }; then ok "$d (exit $rc)"
  else bad "$d (exit $rc, wanted $want)"; tail -n 5 "$SCRATCH/last.out" | sed 's/^/     | /'; fi
}
run() { local img="$1"; shift; docker run --rm --network none --hostname localhost "$@"; }

for img in "$TC" "$TR"; do docker image inspect "$img" >/dev/null 2>&1 || { echo "missing image $img: run build.sh [--truth]" >&2; exit 2; }; done

echo "== (1) tag determinism"
copy="$SCRATCH/tree"; cp -a "$TC_DIR" "$copy"; rm -rf "$copy/tests"
t0="$("$BUILD" --print-tag)"
[ "$("$copy/build.sh" --print-tag)" = "$t0" ] && ok "copy of the tree: same tag" || bad "copy of the tree: tag differs"
find "$copy" -exec touch -d '2001-01-01' {} +
[ "$("$copy/build.sh" --print-tag)" = "$t0" ] && ok "mtimes changed: same tag" || bad "mtimes changed: tag differs"
[ "$(env -i PATH=/usr/bin:/bin "$BUILD" --print-tag)" = "$t0" ] && ok "env -i (the sandbox runner's call shape): same tag" || bad "env -i: tag differs"
[ "$("$BUILD" --truth --print-tag | cut -d: -f2)" = "${t0#*:}" ] && ok "toolchain and truth share one hash" || bad "toolchain and truth hashes differ"
echo x >> "$copy/smoke/java/Main.java"
[ "$("$copy/build.sh" --print-tag)" != "$t0" ] && ok "smoke file edited: tag moves" || bad "smoke file edited: tag did not move"
cp "$TC_DIR/smoke/java/Main.java" "$copy/smoke/java/Main.java"; chmod +x "$copy/smoke/go/main.go"
[ "$("$copy/build.sh" --print-tag)" != "$t0" ] && ok "exec bit changed: tag moves" || bad "exec bit changed: tag did not move"
chmod -x "$copy/smoke/go/main.go"; echo '# x' >> "$copy/Dockerfile"
[ "$("$copy/build.sh" --print-tag)" != "$t0" ] && ok "Dockerfile edited: tag moves" || bad "Dockerfile edited: tag did not move"

echo "== (2) ccg-versions drop-ins"
D="$SCRATCH/dropins"; mkdir -p "$D"
echo 'v bad false'  > "$D/20-bad.sh"; echo 'exit 0' > "$D/40-exit.sh"; echo 'set -e; false' > "$D/30-sete.sh"
for img in "$TC" "$TR"; do
  expect "$img: ccg-versions passes" 0 run "$img" "$img" ccg-versions
  expect "$img: 20-bad + 40-exit (p2-m1) -> non-zero" nonzero run "$img" -v "$D/20-bad.sh:/usr/local/lib/ccg-versions.d/20-bad.sh:ro" -v "$D/40-exit.sh:/usr/local/lib/ccg-versions.d/40-exit.sh:ro" "$img" ccg-versions
  expect "$img: a set -e drop-in that aborts -> non-zero" nonzero run "$img" -v "$D/30-sete.sh:/usr/local/lib/ccg-versions.d/30-sete.sh:ro" "$img" ccg-versions
  expect "$img: a lone exit 0 drop-in -> still passes" 0 run "$img" -v "$D/40-exit.sh:/usr/local/lib/ccg-versions.d/40-exit.sh:ro" "$img" ccg-versions
done
if docker image inspect "$CONTROL" >/dev/null 2>&1; then
  expect "control $CONTROL (pre-S24): 20-bad + 40-exit exits 0 (the bug)" 0 run "$CONTROL" -v "$D/20-bad.sh:/usr/local/lib/ccg-versions.d/20-bad.sh:ro" -v "$D/40-exit.sh:/usr/local/lib/ccg-versions.d/40-exit.sh:ro" "$CONTROL" ccg-versions
else echo "skip control image $CONTROL (not present)"; fi

echo "== (3) the sandbox image carries no truth tool"
expect "toolchain: none of node npm scip scip-python scip-java callgraph uv kotlinc-scip on PATH" 0 \
  run "$TC" "$TC" sh -c 'for c in node npm scip scip-python scip-typescript scip-java scip-clang callgraph uv kotlinc-scip; do command -v $c && exit 1; done; test ! -e /opt/truth'

echo "== (4) smoke"
S="$TC_DIR/smoke"
smoke() {  # smoke <description> <want> <dir> <lang> [extra docker args...]
  local d="$1" want="$2" dir="$3" lang="$4"; shift 4
  expect "$d" "$want" run "$TR" "$@" -v "$dir:/m:ro" -e CCG_TRUTH_SMOKE_DIR=/m "$TR" ccg-truth-smoke "$lang"
}
expect "baseline: ccg-truth-smoke passes (all 8)" 0 run "$TR" "$TR" ccg-truth-smoke
mutant() {  # mutant <lang> <file> <sed-expr> <description>
  local lang="$1" f="$2" e="$3" d="$4" m="$SCRATCH/m-$1-$RANDOM"
  cp -r "$S" "$m"; sed -i "$e" "$m/$lang/$f"
  if cmp -s "$S/$lang/$f" "$m/$lang/$f"; then bad "$lang: mutant '$d' changed nothing (stale sed)"; return; fi
  smoke "$lang: $d -> FAILS" nonzero "$m" "$lang"
}
mutant python     shapes.py   's/class Circle(Shape)/class Circle/'        "override edge removed"
mutant python     shapes.py   's/return shape.area()/return 0.0/'          "call through the base removed"
mutant typescript shapes.ts   's/ implements Shape//'                      "override edge removed"
mutant typescript shapes.ts   's/return shape.area();/return 0;/'          "call through the interface removed"
mutant java       Circle.java 's/ implements Shape//'                      "override edge removed"
mutant java       Main.java   's/return shape.area();/return 0.0;/'        "call through the interface removed"
mutant kotlin     shapes.kt   's/ : Shape//'                               "override edge removed"
mutant kotlin     shapes.kt   's/= shape.area()/= 0.0/'                    "call through the interface removed"
mutant csharp     Shapes.cs   's/ : IShape//'                              "override edge removed"
mutant csharp     Shapes.cs   's/=> shape.Area();/=> 0.0;/'                "call through the interface removed"
mutant cpp        shape.h     's/struct Circle : Shape {/struct Circle {/' "override edge removed"
mutant cpp        shape.cpp   's/sum += s->area();/sum += 1.0;/'           "call through the interface removed"
mutant go         main.go     's/return s.Area()/return 0.0/'              "call through the interface removed"
smoke "cpp: the compiler's builtin headers hidden -> FAILS" nonzero "$S" cpp --tmpfs /usr/lib/gcc/x86_64-linux-gnu/13/include:exec
smoke "cpp: libstdc++ headers hidden -> FAILS" nonzero "$S" cpp --tmpfs /usr/include/c++:exec

echo "== $pass passed, $fail failed"
exit "$fail"
