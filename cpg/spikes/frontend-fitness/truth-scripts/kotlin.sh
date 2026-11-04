#!/usr/bin/env bash
# Kotlin reference truth (Gradle corpus), in the truth image, cwd = writable copy of the corpus (/work/src).
# Why not `scip-java index --build-tool=gradle`: its Gradle plugin injects scip-kotlinc into the project's own Kotlin
# (2.4.20 here) and failed on buildSrc for the first candidate; this path uses Gradle only to build (C4) and to print
# classpaths + source roots, then runs `kotlinc-scip` (S24-proved, embeddable 2.4.21) per module, then `scip-java aggregate`.
set -uo pipefail
cd /work/src
export GRADLE_USER_HOME=/work/gradle-home CCG_OUT=/work/cpdump
export GRADLE_OPTS="-Xmx1g" ORG_GRADLE_PROJECT_dummy=1
G=(./gradlew --no-daemon --console=plain -Dorg.gradle.jvmargs=-Xmx1536m -Pkotlin.compiler.execution.strategy=in-process -Pkotlin.daemon.jvm.options=-Xmx1g)
chmod +x gradlew
# C4: the standard tool builds main + test classes from the clean tree (exit status recorded, not fatal here)
"${G[@]}" assemble testClasses >/work/gradle-build.log 2>&1; echo "gradle assemble testClasses exit=$?" | tee /out/gradle.exit
tail -5 /work/gradle-build.log
"${G[@]}" -I /corpus-truth/kotlin-cp.init.gradle ccgDump >/work/gradle-dump.log 2>&1 || { tail -30 /work/gradle-dump.log; exit 1; }
ls /work/cpdump
# serialization compiler plugin matching the compiler (the project's Gradle plugin version is not usable by 2.4.21)
SER=/work/plugins/kotlin-serialization-compiler-plugin-embeddable-2.4.21.jar
mkdir -p /work/plugins
[ -f "$SER" ] || curl -sfL -o "$SER" https://repo1.maven.org/maven2/org/jetbrains/kotlin/kotlin-serialization-compiler-plugin-embeddable/2.4.21/kotlin-serialization-compiler-plugin-embeddable-2.4.21.jar
rm -rf /work/tgt /work/kout; mkdir -p /work/tgt /work/kout
for cpf in /work/cpdump/*.cp.txt; do
  m="$(basename "$cpf" .cp.txt)"
  mapfile -t roots < /work/cpdump/$m.src.txt
  [ "${#roots[@]}" -gt 0 ] || continue
  XP=(-Xplugin="$SER")   # harmless for modules that do not use @Serializable
  echo "== module $m roots=${#roots[@]} plugins=${#XP[@]}"
  SCIP_SOURCEROOT=/work/src SCIP_TARGETROOT=/work/tgt kotlinc-scip -jvm-target 21 -Xskip-metadata-version-check "${XP[@]}" \
    -cp "$(cat "$cpf")" -d "/work/kout/$m" "${roots[@]}" >"/work/kotlinc-$m.log" 2>&1
  echo "kotlinc-scip $m exit=$?  errors=$(grep -c ': error:' /work/kotlinc-$m.log) warnings=$(grep -c ': warning:' /work/kotlinc-$m.log)"
done
cp /work/kotlinc-*.log /out/ 2>/dev/null || true
scip-java aggregate --targetroot=/work/tgt --output=/out/index.scip
