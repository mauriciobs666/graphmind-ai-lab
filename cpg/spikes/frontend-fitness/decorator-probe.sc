// decorator-probe.sc -- S27c-F4: what node shape does Joern emit for decorators / annotations / attributes? Run ONLY on the planted
// synthetic sources of tests/data/census/real-src/<lang>/ through the runner's `joern --script` form:
//   offline-run.sh --backend container --mode offline --frontend joern --script-file decorator-probe.sc --source <real-src/lang/src> \
//     --output <dir> -- --script /scripts/decorator-probe.sc --param cpgFile=/out/<id>.cpg.bin --param out=/out/@AUDIT_ID@.dprobe.jsonl
// Emits JSONL: {"k":"annotation",...} every ANNOTATION node (+ children, parent), {"k":"mchild",...} the non-BLOCK/PARAMETER AST children
// of each non-external METHOD, {"k":"node",...} every CALL / LITERAL / IDENTIFIER / TYPE_REF / UNKNOWN with file, line, code and AST parent.
// Reads no label or truth file.
import io.shiftleft.codepropertygraph.generated.nodes._
import io.shiftleft.semanticcpg.language._

@main def main(cpgFile: String, out: String, frontendOnly: String = "false") = {
  if (frontendOnly == "true") importCpg(cpgFile, "", false) else importCpg(cpgFile)   // Kotlin: default overlays abort (ReachingDefPass)
  val w = new java.io.PrintWriter(new java.io.OutputStreamWriter(new java.io.FileOutputStream(out), "UTF-8"))
  def emit(o: ujson.Obj): Unit = { w.write(ujson.write(o)); w.write("\n") }
  def oi(o: Option[Any]): Int = o match { case Some(v: java.lang.Integer) => v.intValue; case Some(v: Int) => v; case _ => -1 }
  def lineOf(n: StoredNode): Int = n match { case x: AstNode => oi(x.lineNumber); case _ => -1 }
  def codeOf(n: StoredNode): String = n match { case x: AstNode => x.code.take(100); case _ => "" }
  def fileOf(n: StoredNode): String = scala.util.Try(n match {
    case x: Method => x.filename
    case x: AstNode => x.inAst.collectAll[Method].headOption.map(_.filename).getOrElse("")
    case _ => ""
  }).getOrElse("")
  def sub(n: AstNode): ujson.Value = ujson.Obj("label" -> n.label, "code" -> n.code.take(80), "line" -> oi(n.lineNumber),
    "name" -> (n match { case a: Annotation => a.name; case a: AnnotationParameter => a.code; case _ => "" }))
  cpg.annotation.l.sortBy(a => (fileOf(a), lineOf(a))).foreach { a =>
    val parent = a.astParent
    emit(ujson.Obj("k" -> "annotation", "file" -> fileOf(a), "line" -> lineOf(a), "name" -> a.name, "fullName" -> a.fullName,
      "code" -> a.code.take(120), "parent" -> ujson.Obj("label" -> parent.label, "code" -> codeOf(parent), "line" -> lineOf(parent)),
      "children" -> ujson.Arr.from(a.astChildren.l.map { c =>
        ujson.Obj("label" -> c.label, "code" -> c.code.take(80), "kids" -> ujson.Arr.from(c.astChildren.l.map(sub)))
      })))
  }
  cpg.method.isExternal(false).l.sortBy(m => (m.filename, oi(m.lineNumber))).foreach { m =>
    emit(ujson.Obj("k" -> "mchild", "file" -> m.filename, "name" -> m.name, "l" -> oi(m.lineNumber), "le" -> oi(m.lineNumberEnd),
      "annCount" -> m.annotation.size,
      "children" -> ujson.Arr.from(m.astChildren.l.filterNot(c => c.isInstanceOf[Block] || c.isInstanceOf[MethodParameterIn] || c.isInstanceOf[MethodReturn] || c.isInstanceOf[Modifier]).map(sub))))
  }
  (cpg.call.l ++ cpg.literal.l ++ cpg.identifier.l ++ cpg.typeRef.l ++ cpg.unknown.l).sortBy(n => (fileOf(n), lineOf(n))).foreach { n =>
    val p = n.astParent
    emit(ujson.Obj("k" -> "node", "label" -> n.label, "file" -> fileOf(n), "line" -> lineOf(n), "code" -> codeOf(n),
      "name" -> (n match { case c: Call => c.name; case _ => "" }), "parent" -> ujson.Obj("label" -> p.label, "code" -> codeOf(p))))
  }
  w.flush(); w.close()
}
