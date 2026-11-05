// DS scratch probe: CFG successors and CDG of named methods (read-only).
import io.shiftleft.codepropertygraph.generated.nodes._
import io.shiftleft.semanticcpg.language._
@main def main(cpgFile: String, out: String, targets: String) = {
  importCpg(cpgFile)
  val w = new java.io.PrintWriter(out)
  targets.split(";").foreach { t =>
    val Array(fileSub, mname, lineS) = t.split("@")
    val ms = cpg.method.isExternal(false).l.filter(m => m.filename.contains(fileSub) && m.name == mname && m.lineNumber.contains(lineS.toInt))
    ms.foreach { m =>
      w.println(s"== METHOD ${m.name} ${m.filename}:${m.lineNumber.getOrElse(-1)}")
      m.cfgNode.l.sortBy(n => (n.lineNumber.getOrElse(-1), n.columnNumber.getOrElse(-1))).foreach { n =>
        val succ = n._cfgOut.l.map(s => s"${s.label}@${s.propertiesMap.getOrDefault("LINE_NUMBER", -1)}:${s.propertiesMap.getOrDefault("CODE", "").toString.take(30)}").mkString(" | ")
        val cdg = n._cdgOut.l.map(s => s"${s.propertiesMap.getOrDefault("LINE_NUMBER", -1)}:${s.propertiesMap.getOrDefault("CODE", "").toString.take(30)}").mkString(" | ")
        val cdgIn = n._cdgIn.l.map(s => s"${s.propertiesMap.getOrDefault("LINE_NUMBER", -1)}:${s.propertiesMap.getOrDefault("CODE", "").toString.take(25)}").mkString(" | ")
        w.println(s"  ${n.label}@${n.lineNumber.getOrElse(-1)} [${n.code.take(40)}]  CFG-> ${succ}   CDG-> ${cdg}   CDG<- ${cdgIn}")
      }
    }
  }
  w.close()
}
