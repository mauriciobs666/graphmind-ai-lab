// unmatched-probe.sc -- S27: for each (path, nameLine) listed in /src/unmatched.tsv ("path<TAB>lo<TAB>hi", the frame's 1-based
// line range: callee-expression first line .. name line), print EVERY call (operators included) the CPG holds on that file and Joern-line, and the file's FILE node
// presence. Used only to classify an unmatched labelled site as *matcher error* or *absent in Joern* (ML §7.1).
// Run through the sandbox runner's `joern --script` form like measure.sc. Reads no label or truth.
import io.shiftleft.codepropertygraph.generated.nodes._
import io.shiftleft.semanticcpg.language._

@main def main(cpgFile: String, sites: String, lineOff: Int, out: String) = {
  importCpg(cpgFile)
  val src = scala.io.Source.fromFile(sites, "UTF-8")
  val want = try src.getLines().filter(_.trim.nonEmpty).map { l => val a = l.split("\t"); (a(0), a(1).toInt, a(2).toInt) }.toList finally src.close()
  val w = new java.io.PrintWriter(new java.io.OutputStreamWriter(new java.io.FileOutputStream(out), "UTF-8"))
  def oi(o: Option[Any]): Int = o match { case Some(v: java.lang.Integer) => v.intValue; case Some(v: Int) => v; case _ => -1 }
  val fileNames = cpg.file.name.l.toSet
  want.foreach { case (p, lo, hi) =>
    val cs = cpg.call.filter(c => scala.util.Try(c.method.filename).toOption.contains(p) && oi(c.lineNumber) >= lo + lineOff && oi(c.lineNumber) <= hi + lineOff).l
    val arr = ujson.Arr.from(cs.sortBy(c => (oi(c.columnNumber), c.id)).map { c =>
      ujson.Obj("name" -> c.name, "l" -> oi(c.lineNumber), "c" -> oi(c.columnNumber), "code" -> c.code.take(100), "dispatch" -> c.dispatchType) })
    w.write(ujson.write(ujson.Obj("k" -> "probe", "path" -> p, "lo" -> lo, "hi" -> hi, "fileNode" -> fileNames.contains(p), "calls" -> arr)) + "\n")
  }
  w.flush(); w.close()
}
