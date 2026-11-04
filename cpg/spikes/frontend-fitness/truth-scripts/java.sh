#!/usr/bin/env bash
# Java reference truth (Maven corpus): scip-java's manual path (plugin on javac, then `aggregate`), because the
# automatic `scip-java index` supports Gradle only. Runs in the truth image, cwd = writable copy of the corpus.
set -euo pipefail
cd /work/src
export MAVEN_OPTS="-Xmx1g"
mvn -B -ntp -q -DskipTests generate-sources >/work/mvn-gen.log 2>&1 || { tail -20 /work/mvn-gen.log; exit 1; }
mvn -B -ntp -q dependency:build-classpath -Dmdep.outputFile=/work/cp.txt -Dmdep.includeScope=test >/work/mvn-cp.log 2>&1 || { tail -20 /work/mvn-cp.log; exit 1; }
CP="$(cat /work/cp.txt)"
rm -rf /work/tgt /work/classes; mkdir -p /work/tgt /work/classes
{ find src/main/java src/test/java -name '*.java' 2>/dev/null; find target/generated-sources -name '*.java' 2>/dev/null; } | sort > /work/sources.txt
wc -l /work/sources.txt
javac --release 17 -parameters -proc:none -cp "$SCIP_JAVAC_JAR:$CP" -d /work/classes \
  "-Xplugin:scip -sourceroot:/work/src -targetroot:/work/tgt" @/work/sources.txt
scip-java aggregate --targetroot=/work/tgt --output=/out/index.scip
