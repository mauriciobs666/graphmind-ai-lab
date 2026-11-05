// DS scratch probe: reaching-def sources and AST parents of named identifiers on given lines (read-only).
import io.shiftleft.codepropertygraph.generated.nodes._
import io.shiftleft.semanticcpg.language._
@main def main(cpgFile: String, out: String, targets: String) = {
  importCpg(cpgFile)
  val w = new java.io.PrintWriter(out)
  def d(n: StoredNode): String = s"${n.label}@${n.propertiesMap.getOrDefault("LINE_NUMBER", -1)}[${n.propertiesMap.getOrDefault("CODE", "").toString.replace("\n"," ").take(50)}]"
  targets.split(";").foreach { t =>
    val Array(fileSub, lineS, name) = t.split("@")
    val ids = cpg.identifier.l.filter(i => i.file.name.headOption.exists(_.contains(fileSub)) && i.lineNumber.contains(lineS.toInt) && i.name == name)
    w.println(s"== $t : ${ids.size} identifiers")
    ids.foreach { i =>
      val chain = Iterator.iterate(i: AstNode)(_.astParent).drop(1).take(3).map(n => d(n)).mkString(" <- ")
      w.println(s"  id ${i.id} argIdx=${i.argumentIndex} parents: $chain")
      w.println(s"    rdIn: " + i._reachingDefIn.l.map(x => d(x)).mkString(" | "))
      w.println(s"    rdOut: " + i._reachingDefOut.l.map(x => d(x)).take(6).mkString(" | "))
    }
  }
  w.close()
}
