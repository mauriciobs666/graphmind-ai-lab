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
//   {"k":"meta", cpg, lang, files, methods (non-external), calls (non-operator), callsWithoutMethod, frameSites, censusSchema, frontendOnly}
//   {"k":"file", path, methods}                    every FILE node (methods = non-external METHODs in it)
//   {"k":"method", file, name, fullName, l, le, vendored, ann[]}   every non-external METHOD (line fidelity / coverage joins)
//     ann[]  = {name, node, code, lits[], litNodes[]} per annotation / decorator / attribute of the METHOD (S27c census, censusSchema 1):
//              node ANNOTATION = an ANNOTATION node on the METHOD (java, kotlin, csharp, javascript/typescript); lits = the `code` of every
//              ANNOTATION_LITERAL / LITERAL under it (litNodes = their labels). node CALL = a Python decorator, recovered from the
//              frontend's rewrite `f = deco(<f's METHOD_REF>)` / `f = obj.deco(args)(<METHOD_REF>)` (name = the decorator call's `name`,
//              lits = its LITERAL arguments and those inside a `<operator>.listLiteral` argument). Frontends that emit no node
//              (go, cpp attributes) give ann = [] (decorator-probe.sc, results/<lang>.decorator-probe.json).
//   {"k":"site", key, matched, call{name,l,c,code}, res[], nm}   every frame site; for SAMPLED sites also nmList[]
//     res[]  = the matched CALL's resolved targets (CALL edges), each {fullName,file,l,vendored,ext}
//     nm     = number of in-corpus (non-external, non-vendored) METHODs whose name equals the site's calleeName
//     nmList = those METHODs {file,l,fullName} (sampled sites only; the census needs only the counts)
//   {"k":"carg", key, order, recv, kind, lit, nreach, reach[]}   (S27c) one per argument of each MATCHED site's call, right after its site record:
//     order  = 1-based rank of the argument among the call's `argument`s sorted by (argumentIndex, node id), receiver included when present.
//              Exception (observed, pysrc2cpg): a keyword argument `k=v` carries argumentIndex -1, which would rank it first; when any
//              argument has a negative argumentIndex the whole list is ranked by AST `order` (source order) instead.
//     recv   = the call has an argument with argumentIndex 0, Joern's receiver slot (observed: java, csharp, cpp put the RECEIVER edge on it;
//              python and javascript put it on the callee expression instead and still carry the instance as argumentIndex 0; javascript
//              adds an implicit `this` at 0 to a plain call; go and plain python/cpp calls have none), so rank 1 is the receiver; per call
//     ai     = the raw argumentIndex; aname = argumentName when set
//     kind   = the argument node's label; lit = Literal `code` when kind is LITERAL else null
//     nreach = distinct reaching definitions (REACHING_DEF edges into an IDENTIFIER argument, an assignment's literal and its target identifier
//              counted once; the METHOD entry node and sibling arguments of the call are not definitions); 0 when the dataflow overlay is absent; reach = the `code` of each such definition that is a Literal
//   --param frontendOnly=true imports the CPG without the default overlays (Kotlin: ReachingDefPass aborts them; meta.frontendOnly = true).
// Matching (ML §7.1): same file, Joern call line == frame nameLine, call name accepted by matcher-table.md for the
// site kind, nearest column to the callee-expression start wins (a call with no column sorts last), then node id.
// matcher-table.md is the one place the rules live; `nameOk` below is its transcription.
import io.shiftleft.codepropertygraph.generated.nodes._
import io.shiftleft.semanticcpg.language._
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

@main def main(cpgFile: String, frame: String, sample: String, lang: String, vendored: String, out: String, frontendOnly: String = "false") = {
  val fo = frontendOnly == "true"
  if (fo) importCpg(cpgFile, "", false) else importCpg(cpgFile)
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

  // ---- census (S27c): annotations / decorators per METHOD, call arguments per matched site -------------------------------
  def annNodeOf(a: Annotation): ujson.Obj = {
    val lits = ArrayBuffer[String](); val kinds = ArrayBuffer[String]()
    def walk(n: AstNode): Unit = n.astChildren.l.foreach {
      case l: AnnotationLiteral => lits.append(l.code); kinds.append("ANNOTATION_LITERAL")
      case l: Literal => lits.append(l.code); kinds.append("LITERAL")
      case c => walk(c)
    }
    walk(a)
    ujson.Obj("name" -> a.name, "node" -> "ANNOTATION", "code" -> a.code.take(200), "lits" -> ujson.Arr.from(lits), "litNodes" -> ujson.Arr.from(kinds))
  }
  // Python: pysrc2cpg rewrites `@d def f` to `f = d(<METHOD_REF f>)` and `@o.d(a) def f` to `f = o.d(a)(<METHOD_REF f>)`
  // (a dynamic-dispatch call with name "" whose receiver is the decorator call, or a BLOCK ending in it). Climb from the METHOD_REF.
  lazy val mrefByFull: Map[String, List[MethodRef]] = cpg.methodRef.l.groupBy(_.methodFullName)
  def pyDecorators(m: Method): List[ujson.Obj] = {
    val found = ArrayBuffer[(Int, Int, ujson.Obj)]()
    mrefByFull.getOrElse(m.fullName, Nil).foreach { mr =>
      var cur: AstNode = mr
      var p: AstNode = mr.astParent
      while (p.isInstanceOf[Call] && !p.asInstanceOf[Call].name.startsWith("<operator>") &&
             p.asInstanceOf[Call].argument.l.exists(_.id == cur.id)) {
        val pc = p.asInstanceOf[Call]
        val deco: Option[Call] =
          if (pc.name.nonEmpty) Some(pc)
          else pc.receiver.l.headOption match {
            case Some(c: Call) => Some(c)
            case Some(b: Block) => b.astChildren.l.collect { case c: Call => c }.lastOption
            case _ => None
          }
        deco.foreach { d =>
          val lits = ArrayBuffer[String]()
          d.argument.l.foreach {
            case l: Literal => lits.append(l.code)
            case c: Call if c.name == "<operator>.listLiteral" => c.argument.l.foreach { case l: Literal => lits.append(l.code); case _ => }
            case _ =>
          }
          val code = if (d.id == pc.id) d.name else d.code.take(200)
          found.append((oi(d.lineNumber), oi(d.columnNumber), ujson.Obj("name" -> d.name, "node" -> "CALL", "code" -> code,
            "lits" -> ujson.Arr.from(lits), "litNodes" -> ujson.Arr.from(lits.map(_ => "LITERAL")))))
        }
        cur = p; p = p.astParent
      }
    }
    found.toList.distinct.sortBy(x => (x._1, x._2)).map(_._3)
  }
  def annOf(m: Method): ujson.Arr = {
    val a = m.annotation.l.sortBy(x => (oi(x.lineNumber), oi(x.columnNumber), x.id)).map(annNodeOf)
    ujson.Arr.from(if (lang == "python") a ++ pyDecorators(m) else a)
  }
  // Reaching definitions of an IDENTIFIER argument -> (distinct definition keys, Literal codes among them).
  // Sources that are arguments of the same call are excluded: with no semantics for the callee, the dataflow engine routes every argument into
  // every other (observed: `call("s", LIMIT)` gives LIMIT the definition "s").
  def reachOf(a: Expression, call: Call): (Int, List[String]) = a match {
    case i: Identifier =>
      val srcs = i._reachingDefIn.l.filterNot { s => s match {
        case _: Method => true                                    // the METHOD entry node is a source of every use (observed, pysrc2cpg): not a definition
        case n: AstNode => n.astParent.id == call.id
        case _ => false } }
      val keyed = srcs.map { s =>
        val asg: Option[Call] = s match {
          case n: AstNode => n.astParent match { case c: Call if c.name == "<operator>.assignment" => Some(c); case _ => None }
          case _ => None
        }
        val litCode: Option[String] = (s, asg) match {
          case (l: Literal, _) => Some(l.code)
          case (_, Some(c)) => c.argument.l.find(_.argumentIndex == 2).collect { case l: Literal => l.code }
          case _ => None
        }
        (asg.map(_.id).getOrElse(s.id), litCode)
      }.distinctBy(_._1)
      (keyed.size, keyed.flatMap(_._2))
    case _ => (0, Nil)
  }

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
    "methods" -> methods.size, "calls" -> calls.size, "callsWithoutMethod" -> noMethod, "frameSites" -> frameRows.size,
    "censusSchema" -> 1, "frontendOnly" -> fo))
  cpg.file.name.l.distinct.sorted.foreach { fn => emit(ujson.Obj("k" -> "file", "path" -> fn, "methods" -> byFile.getOrElse(fn, Nil).size)) }
  methods.sortBy(m => (m.filename, mLine(m), m.fullName)).foreach { m =>
    emit(ujson.Obj("k" -> "method", "file" -> m.filename, "name" -> m.name, "fullName" -> m.fullName,
      "l" -> mLine(m), "le" -> mEnd(m),
      "vendored" -> isVendored(m.filename), "ann" -> annOf(m)))
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
    best.foreach { c =>
      val args0 = c.argument.l
      val args = if (args0.exists(_.argumentIndex < 0)) args0.sortBy(a => (a.order, a.id)) else args0.sortBy(a => (a.argumentIndex, a.id))
      val recv = args.exists(_.argumentIndex == 0)
      args.zipWithIndex.foreach { case (a, i) =>
        val (nreach, reach) = reachOf(a, c)
        val ao = ujson.Obj("k" -> "carg", "key" -> key, "order" -> (i + 1), "recv" -> recv, "ai" -> a.argumentIndex, "kind" -> a.label,
          "lit" -> (a match { case l: Literal => ujson.Str(l.code); case _ => ujson.Null }),
          "nreach" -> nreach, "reach" -> ujson.Arr.from(reach))
        a.argumentName.foreach(n => ao("aname") = n)
        emit(ao)
      }
    }
  }
  w.flush(); w.close()
}
