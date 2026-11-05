// measure.sc -- S25 (P0-18b) Joern-side measurement of ONE CPG against the language's census frame.
//
// Run ONLY through the sandbox runner's `joern --script` form (egress spec §7.3.1; reviewed as code at S25's gate,
// egress §11 R10). The runner stages this file at /scripts/measure.sc (hashed into the audit record), mounts the
// staged input directory read-only at /src and the output directory at /out:
//
//   cpg/sandbox/offline-run.sh --backend container --mode offline --frontend joern \
//     --script-file cpg/spikes/frontend-fitness/measure.sc --source <stage dir> --output <dir> \
//     -- --script /scripts/measure.sc --param cpgFile=/out/<parse auditId>.cpg.bin \
//        --param frame=/src/frame.jsonl --param sample=/src/sample.private.jsonl --param lang=<lang> \
//        --param vendored=vendor,vendored,_vendor,third_party,third-party,node_modules \
//        --param out=/out/@AUDIT_ID@.measure.jsonl
//   (<stage dir> holds frame/<lang>.jsonl as frame.jsonl and sample/<lang>.private.jsonl as sample.private.jsonl
//    only: neither carries a label or reference truth; run_fitness.py `joern` prints the exact lines.)
//
// It reads NO label, NO reference truth and NO adjudication file: it only asks the CPG its own answers. Output is
// JSONL, one record per line, ordered (deterministic given the CPG):
//   {"k":"meta", cpg, lang, files, methods (non-external), calls (non-operator), frameSites}
//   {"k":"file", path, methods}                    every FILE node (methods = non-external METHODs in it)
//   {"k":"method", file, name, fullName, l, le, vendored}   every non-external METHOD (line fidelity / coverage joins)
//   {"k":"site", key, matched, call{name,l,c,code}, res[], nm}   every frame site; for SAMPLED sites also nmList[]
//     res[]  = the matched CALL's resolved targets (CALL edges), each {fullName,file,l,vendored,ext}
//     nm     = number of in-corpus (non-external, non-vendored) METHODs whose name equals the site's calleeName
//     nmList = those METHODs {file,l,fullName} (sampled sites only; the census needs only the counts)
// Matching (ML §7.1): same file, Joern call line == frame nameLine, call name accepted by matcher-table.md for the
// site kind, nearest column to the callee-expression start wins (a call with no column sorts last), then node id.
// matcher-table.md is the one place the rules live; `nameOk` below is its transcription.
import io.shiftleft.codepropertygraph.generated.nodes._
import io.shiftleft.semanticcpg.language._
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

@main def main(cpgFile: String, frame: String, sample: String, lang: String, vendored: String, out: String) = {
  importCpg(cpgFile)
  val vend = vendored.split(",").map(_.trim).filter(_.nonEmpty).toSet
  def isVendored(p: String): Boolean = p.split("/").dropRight(1).exists(vend.contains)
  def readJsonl(p: String): List[ujson.Value] = {
    val src = scala.io.Source.fromFile(p, "UTF-8")
    try src.getLines().filter(_.trim.nonEmpty).map(ujson.read(_)).toList finally src.close()
  }
  val w = new java.io.PrintWriter(new java.io.OutputStreamWriter(new java.io.FileOutputStream(out), "UTF-8"))
  def emit(o: ujson.Obj): Unit = { w.write(ujson.write(o)); w.write("\n") }
  def oi(o: Option[Any]): Int = o match { case Some(v: java.lang.Integer) => v.intValue; case Some(v: Int) => v; case _ => -1 }
  def lineOf(c: Call): Int = oi(c.lineNumber)
  def colOf(c: Call): Int = oi(c.columnNumber)
  def mLine(m: Method): Int = oi(m.lineNumber)
  def mEnd(m: Method): Int = oi(m.lineNumberEnd)
  def isOp(c: Call): Boolean = c.name.startsWith("<operator>") || c.name.startsWith("<operators>")

  // ---- matcher (matcher-table.md, observed on synthetic probes with `matcher-probe.sc`) -----------------------
  // Does CALL `c` correspond to a frame site of `kind` whose callee's last identifier is `callee`?
  def nameOk(kind: String, callee: String, c: Call): Boolean = {
    val n = c.name
    (lang, kind) match {
      case ("javascript", "ctor")  => n == "<operator>.new" && c.code.startsWith("new ") && c.code.contains(callee)
      // optional-chain call `a?.b(x)`: jssrc2cpg names the CALL with the whole callee text (`a?.b`), found by S27's
      // unmatched-site inspection (matcher-table.md "JS/TS optional chain"); the code must begin with that name.
      case ("javascript", _)       => n == callee || (n.endsWith("?." + callee) && c.code.startsWith(n))
      case ("java" | "csharp", "ctor") => n == "<init>" || n == callee
      case ("kotlin", _)           => n == callee || n == "<init>"                       // `Foo(3)` is a plain-looking frame site
      case ("cpp", "ctor")         => n == callee
      case ("cpp", _)              => n == callee || n.startsWith(callee + "<")          // c2cpg keeps template args in the name
      case _                       => n == callee
    }
  }

  // Joern's positions are NOT uniform (observed on probes): python, java, go, cpp lines+columns are 1-based like the
  // frame; javascript and kotlin columns are 0-based; csharp lines AND columns are 0-based.
  val colOff = if (Set("javascript", "kotlin", "csharp").contains(lang)) -1 else 0
  val lineOff = if (lang == "csharp") -1 else 0

  val methods = cpg.method.isExternal(false).l
  val byFile = methods.groupBy(_.filename)
  // in-corpus name index (non-vendored)
  val byName = mutable.HashMap[String, ArrayBuffer[Method]]()
  methods.filterNot(m => isVendored(m.filename)).foreach(m => byName.getOrElseUpdate(m.name, ArrayBuffer()).append(m))

  // ---- calls indexed by (file, line) -------------------------------------------------------------------------
  val calls = cpg.call.filter(c => !isOp(c) || c.name == "<operator>.new").l
  val callIdx = mutable.HashMap[(String, Int), ArrayBuffer[Call]]()
  // csharpsrc2cpg leaves some calls without an enclosing METHOD (`c.method` is null: NPE seen on Ocelot); they cannot be
  // attributed to a file, so they are skipped (counted in meta.callsWithoutMethod).
  var noMethod = 0
  calls.foreach { c =>
    val fn = scala.util.Try(c.method.filename).toOption
    if (fn.isEmpty) noMethod += 1 else callIdx.getOrElseUpdate((fn.get, lineOf(c)), ArrayBuffer()).append(c)
  }

  val frameRows = readJsonl(frame)
  val sampled = readJsonl(sample).map(_("siteKey").str).toSet

  emit(ujson.Obj("k" -> "meta", "cpg" -> cpgFile, "lang" -> lang, "files" -> cpg.file.name.l.distinct.size,
    "methods" -> methods.size, "calls" -> calls.size, "callsWithoutMethod" -> noMethod, "frameSites" -> frameRows.size))
  cpg.file.name.l.distinct.sorted.foreach { fn => emit(ujson.Obj("k" -> "file", "path" -> fn, "methods" -> byFile.getOrElse(fn, Nil).size)) }
  methods.sortBy(m => (m.filename, mLine(m), m.fullName)).foreach { m =>
    emit(ujson.Obj("k" -> "method", "file" -> m.filename, "name" -> m.name, "fullName" -> m.fullName,
      "l" -> mLine(m), "le" -> mEnd(m),
      "vendored" -> isVendored(m.filename)))
  }

  frameRows.foreach { r =>
    val key = r("siteKey").str
    val path = r("path").str
    val nameLine = r("nameLine").num.toInt + lineOff
    val callee = r("calleeName").str
    val startCol = r("column").num.toInt + colOff   // the callee expression's first character: Joern's call column
    val kind = r("kind").str
    // Fallback found by S27's unmatched-site inspection (matcher-table.md "multi-line chains"): when nothing on the name's
    // line matches, accept a call on the line of the callee expression's first character (javasrc2cpg, gosrc2cpg put a
    // chained or composite-literal call's line there, not on the name's line).
    val exprLine = r("line").num.toInt + lineOff
    val cands0 = callIdx.getOrElse((path, nameLine), ArrayBuffer()).filter(c => nameOk(kind, callee, c)).toList
    val cands = if (cands0.nonEmpty || exprLine == nameLine) cands0
                else callIdx.getOrElse((path, exprLine), ArrayBuffer()).filter(c => nameOk(kind, callee, c)).toList
    // Joern gives some calls no column (-1, e.g. Java `<init>`): those sort after calls with a column
    val best = cands.sortBy(c => (if (colOf(c) < 0) Int.MaxValue else math.abs(colOf(c) - startCol), c.id)).headOption
    val nmAll = byName.getOrElse(callee, ArrayBuffer()).toList
    val res: List[Method] = best.toList.flatMap(_._callOut.collect { case m: Method => m }.toList).distinct
    val resJ = res.sortBy(m => (m.filename, mLine(m), m.fullName)).map { m =>
      ujson.Obj("fullName" -> m.fullName, "file" -> m.filename, "l" -> mLine(m),
        "vendored" -> isVendored(m.filename), "ext" -> m.isExternal)
    }
    val o = ujson.Obj("k" -> "site", "key" -> key, "matched" -> best.isDefined, "nm" -> nmAll.size,
      "res" -> ujson.Arr.from(resJ))
    best.foreach { c => o("call") = ujson.Obj("name" -> c.name, "l" -> lineOf(c), "c" -> colOf(c), "code" -> c.code.take(120)) }
    if (sampled.contains(key)) {
      o("nmList") = ujson.Arr.from(nmAll.sortBy(m => (m.filename, mLine(m))).map { m =>
        ujson.Obj("file" -> m.filename, "l" -> mLine(m), "fullName" -> m.fullName)
      })
    }
    emit(o)
  }
  w.flush(); w.close()
}
