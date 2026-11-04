// matcher-probe.sc -- S25 (ML §7.1): lists every CALL of a CPG with the fields the site matcher keys on
// (name, code, line, column, enclosing method, resolved callee full names), so matcher-table.md is written
// from observation. Run on SYNTHETIC probe projects only (never a corpus), through the runner's script form:
//   offline-run.sh --backend container --mode offline --frontend joern --script-file matcher-probe.sc \
//     --source <probe dir> --output <dir> -- --script /scripts/matcher-probe.sc --param cpgFile=/out/<id>.cpg.bin
import io.shiftleft.codepropertygraph.generated.nodes._
import io.shiftleft.semanticcpg.language._

@main def main(cpgFile: String) = {
  importCpg(cpgFile)
  def oi(o: Option[Any]): Int = o match { case Some(v: java.lang.Integer) => v.intValue; case Some(v: Int) => v; case _ => -1 }
  cpg.call.l.sortBy(c => (oi(c.lineNumber), oi(c.columnNumber), c.name)).foreach { c =>
    val callees = c.callee.l.map(m => m.fullName + (if (m.isExternal) "[ext]" else "")).mkString("|")
    val nm = c.name
    if (!nm.startsWith("<operator>.assignment") && !nm.startsWith("<operator>.fieldAccess") && !nm.startsWith("<operator>.indirectFieldAccess") && !nm.startsWith("<operator>.addition") && !nm.startsWith("<operator>.formatString") && !nm.startsWith("<operator>.indexAccess"))
      println("CALL\t" + nm + "\t" + oi(c.lineNumber) + ":" + oi(c.columnNumber) + "\t" + c.code.take(60).replace("\n", " ") + "\t" + c.dispatchType + "\t" + c.methodFullName + "\t" + callees)
  }
}
