#!/usr/bin/env bash
# C/C++ reference truth: CMake configure + BUILD with the image's gcc 13 (no system clang in the truth image),
# which emits compile_commands.json; scip-clang (bundled Clang/LLVM 21.1.8) then indexes it from the project
# root with --jobs 2.  Decision record: README.md "C/C++: gcc compilation database, not a clang build".
set -euo pipefail
cd /work/src
# git submodules are absent from a `git archive` tree: fetch what the corpus's CMake expects (truth run only,
# in the work copy; the labeller's tree never has these)
if [ -f .gitmodules ]; then
  while read -r path url; do
    [ -z "$(ls -A "$path" 2>/dev/null)" ] && git clone -q --depth 1 "$url" "$path" || true
  done < <(awk '/path =/{p=$3} /url =/{print p, $3}' .gitmodules)
fi
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_STANDARD=17 \
  ${CCG_CMAKE_ARGS:-} >/work/cmake.log 2>&1 || { tail -30 /work/cmake.log; exit 1; }
cmake --build build -j 2 >/work/build.log 2>&1 || { tail -30 /work/build.log; exit 1; }
echo "build ok: $(grep -c '"command"' build/compile_commands.json) compile commands"
scip-clang --compdb-path build/compile_commands.json --jobs 2 --index-output-path /out/index.scip \
  --show-compiler-diagnostics 2>&1 | tail -25
