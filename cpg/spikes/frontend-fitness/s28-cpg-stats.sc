// S28: whole-CPG node / edge / file / method counts (descriptive density, ML §7.3 "nodes and edges per file"). Read-only.
// Run only through the sandbox runner: --script /scripts/s28-cpg-stats.sc --param cpgFile=/out/<cpg>.cpg.bin
import io.shiftleft.semanticcpg.language._
@main def main(cpgFile: String) = {
  importCpg(cpgFile)
  val nodes = cpg.graph.allNodes.size
  val edges = cpg.graph.allNodes.foldLeft(0L)((a, n) => a + n.outE.size)
  val files = cpg.file.name.l.filterNot(n => n.startsWith("<")).size
  println(s"CPGSTATS nodes=$nodes edges=$edges files=$files methods=${cpg.method.isExternal(false).size} calls=${cpg.call.size}")
}
