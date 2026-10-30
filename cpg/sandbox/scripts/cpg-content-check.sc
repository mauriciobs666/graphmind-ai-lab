// cpg-content-check.sc — the one CPG content check for C-1001 P0 (plan §2 D15, P3-7; egress §11 R11).
// joern-parse can exit 0 with an empty CPG (egress §9 H12), so a parse is certified by content.
// Run it only through the runner's script mode (egress §7.3.1):
//   cpg/sandbox/offline-run.sh --backend container --mode offline --frontend joern \
//     --script-file cpg/sandbox/scripts/cpg-content-check.sc --source <dir> --output <cpg dir> \
//     -- --script /scripts/cpg-content-check.sc --param 'cpgFiles=/out/<a>.cpg.bin;/out/<b>.cpg.bin'
// For each CPG (';'-separated) it prints, in this order:
//   METHODS <cpg>=<sorted distinct non-external METHOD names, comma-joined>
//   METHOD_COUNT <cpg>=<number of non-external METHOD nodes>
//   FILE_METHODS <cpg>\t<file>=<non-external METHOD nodes whose filename is <file>>   (one per FILE)
//   CONTENT <cpg>=ok|EMPTY
// The verdict is the unit-level floor: EMPTY iff the CPG has no non-external METHOD. The per-file
// counts are reported, not judged; whether a per-file rule holds is S11's to measure (D15).
// Exit status: 1 if any CPG is EMPTY, else 0. Consumers still read the CONTENT lines.
// Owner: S08 (devops). S11, S25, S28 and S39 run it; none edits it (plan §5.2).
@main def main(cpgFiles: String) = {
  var empty = 0
  cpgFiles.split(";").filter(_.nonEmpty).foreach { f =>
    importCpg(f)
    val ms = cpg.method.isExternal(false).l
    println(s"METHODS $f=" + ms.map(_.name).distinct.sorted.mkString(","))
    println(s"METHOD_COUNT $f=${ms.size}")
    val byFile = ms.groupBy(_.filename).view.mapValues(_.size).toMap
    cpg.file.name.l.distinct.sorted.foreach { fn => println(s"FILE_METHODS $f\t$fn=${byFile.getOrElse(fn, 0)}") }
    val ok = ms.nonEmpty
    if (!ok) empty += 1
    println(s"CONTENT $f=" + (if (ok) "ok" else "EMPTY"))
  }
  if (empty > 0) { System.out.flush(); sys.exit(1) }
}
