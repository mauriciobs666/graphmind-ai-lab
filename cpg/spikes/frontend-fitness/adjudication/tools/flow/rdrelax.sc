// extract.sc — S11 (P0-10) Joern extractor for the connected-code-graph vertical slice.
//
// Contract: docs/plans/connected-code-graph-p0-2.md §4.4 (S11), bundle spec v1.3 + -spec2 v1.3.
// Run ONLY through the sandbox runner, in egress §7.3.1's `joern --script` form (runner-staged,
// hashed into sandbox.script.sha256, no `//>` directive anywhere in this file):
//
//   cpg/sandbox/offline-run.sh --backend container --mode offline --frontend joern \
//     --script-file cpg/spikes/vertical-slice/extract.sc --source <staged unit> --output <dir> \
//     -- --script /scripts/extract.sc --param cpgFile=/out/<parse auditId>.cpg.bin \
//        --param out=/out/@AUDIT_ID@.facts.jsonl --param lang=python|javascript
//
// What it does: for every non-external METHOD of the CPG it writes one JSONL "facts" file that
// normalize.py turns into a bundle. It emits CPG facts only (ids are Joern node ids; bundle ids
// are computed by normalize.py with ccg.ids):
//   m    method (name, fullName, span, AST parent)         t   type decl
//   p    parameter (Joern index)                           r   return statement
//   c    condition node (a CDG source)                     cs  call site (+ argument trees)
//   deco decorator application (Python lowering)          const  module/type-level assignment
//   d    data record candidates: THE PER-METHOD PROJECTION (REACHING_DEF walked backward from each
//        use point -- a call-site argument, a return, a condition -- through identifiers and
//        operators to the nearest point definitions: a parameter or a call result)
//   cb   controlledBy candidates: CDG walked from each point's statement to its condition nodes.
//        RAW: self-edges and repeated (point, cond) pairs are KEPT here and counted in the
//        "stats" record; normalize.py drops them (G-4, -spec2 §1.4) so the CPG shows they occur.
//   stats  timings (load, projection) and the per-method projection-time histogram
//
// Expression trees ("e"): {"o":"lit","c":code} {"o":"name","n":id} {"o":"attr","b":e,"f":field}
// {"o":"fmt","p":[e..]} {"o":"add","p":[e..]} {"o":"call","n":name} {"o":"unk","l":label}
// Every output order is by (line, column, id) or by Joern's AST `order`; nothing depends on the
// order Joern returns nodes in (normalize.py re-sorts by the spec's keys anyway).
import io.shiftleft.codepropertygraph.generated.nodes._
import io.shiftleft.semanticcpg.language._
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

@main def main(cpgFile: String, out: String, lang: String) = {
  val tStart = System.nanoTime
  importCpg(cpgFile)
  val tLoaded = System.nanoTime
  val w = new java.io.PrintWriter(new java.io.OutputStreamWriter(new java.io.FileOutputStream(out), "UTF-8"))
  def emit(o: ujson.Obj): Unit = { w.write(ujson.write(o)); w.write("\n") }

  def ip(n: StoredNode, k: String): Int = n.propertiesMap.get(k) match {
    case i: java.lang.Integer => i.intValue
    case _ => -1
  }
  def sp(n: StoredNode, k: String): String = n.propertiesMap.get(k) match {
    case s: String => s
    case _ => ""
  }
  def line(n: StoredNode): Int = ip(n, "LINE_NUMBER")
  def col(n: StoredNode): Int = ip(n, "COLUMN_NUMBER")
  def code(n: StoredNode): String = sp(n, "CODE")
  def isOp(c: Call): Boolean = c.name.startsWith("<operator>") || c.name.startsWith("<operators>")
  def opName(c: Call): String = c.name.stripPrefix("<operator>.")

  // ---- own-scope walk: AST descendants in `order`, never entering a nested METHOD / TYPE_DECL ----
  def kids(n: AstNode): List[AstNode] = n.astChildren.l.sortBy(x => (x.order, x.id))
  def ownWalk(n: AstNode): Iterator[AstNode] =
    Iterator.single(n) ++ kids(n).iterator.filter(k => !k.isInstanceOf[Method] && !k.isInstanceOf[TypeDecl]).flatMap(ownWalk)

  // Named (keyword) arguments carry ARGUMENT_NAME and argumentIndex -1 in Joern's model.
  def isArg(a: Expression): Boolean = a.argumentIndex >= 0 || a.argumentName.exists(_.nonEmpty)
  def isPos(a: Expression): Boolean = a.argumentIndex >= 1 || a.argumentName.exists(_.nonEmpty)

  // ---- assignment helpers ----
  def isAssign(c: Call): Boolean = c.name.startsWith("<operator>.assignment")
  def arg(c: Call, i: Int): Option[Expression] = c.argument.l.find(_.argumentIndex == i)
  def fieldOf(e: Expression): Option[String] = e match {
    case f: FieldIdentifier => Some(f.canonicalName)
    case _ => None
  }
  def litKey(e: Expression): Option[String] = e match {
    case l: Literal =>
      val s = l.code
      if (s.length >= 2 && (s.head == '"' || s.head == '\'') && s.last == s.head) Some(s.substring(1, s.length - 1)) else None
    case _ => None
  }

  // ---- object / dict literal lowering: BLOCK { tmp = {}; tmp.f = v; tmp["k"] = v; ...; tmp } ----
  // returns the (key, value-expression) pairs, in order, or None when `b` is not that shape.
  def objFields(b: Block): Option[List[(String, Expression)]] = {
    val cs = kids(b).collect { case e: Expression => e }
    if (cs.isEmpty) return None
    val assigns = cs.collect { case c: Call if isAssign(c) => c }
    if (assigns.isEmpty) return None
    // Python: `tmp = {}` first; JS/TS: LOCAL tmp, then `tmp.f = v` ..., and the block's value is `tmp` (last child)
    def initName(c: Call): Option[String] = (arg(c, 1), arg(c, 2)) match {
      case (Some(i: Identifier), Some(r: Call)) if Set("dictLiteral", "objectLiteral", "listLiteral", "alloc").contains(opName(r)) => Some(i.name)
      case _ => None
    }
    val tmp: String = initName(assigns.head).orElse(cs.last match { case i: Identifier => Some(i.name); case _ => None }) match {
      case Some(t) => t
      case None => return None
    }
    val fs = ArrayBuffer[(String, Expression)]()
    assigns.foreach { a =>
      (arg(a, 1), arg(a, 2)) match {
        case (Some(t: Call), Some(v)) if opName(t) == "fieldAccess" || opName(t) == "indexAccess" =>
          (arg(t, 1), arg(t, 2)) match {
            case (Some(base: Identifier), Some(k)) if base.name == tmp =>
              val key = if (opName(t) == "fieldAccess") fieldOf(k) else litKey(k)
              key.foreach(kk => fs += ((kk, v)))
            case _ =>
          }
        case _ =>
      }
    }
    if (fs.isEmpty && initName(assigns.head).isEmpty) None else Some(fs.toList)
  }

  // ---- decorator application detection (Python lowering: `f = deco(METHOD_REF)` chains) ----
  // The chain calls are application plumbing, not call sites (bundle §5.6: decorators are records).
  val decoSkip = mutable.Set[Long]()

  def decoChain(e: Expression): Option[(List[Call], MethodRef)] = e match {
    case r: MethodRef => Some((Nil, r))
    case c: Call if !isOp(c) =>
      val a = c.argument.l.filter(_.argumentIndex == 1)
      if (a.size == 1) decoChain(a.head).map { case (cs, r) => (c :: cs, r) } else None
    case _ => None
  }

  // Python's frontend leaves METHOD_REF children on calls whose "function" is an attribute (type-recovery
  // artifacts, e.g. `list(shop.locales)`); a real function value passed as an argument is a METHOD_REF to a
  // method nested in the current one.
  var curFull: String = ""
  def isArtifactRef(a: AstNode): Boolean = a match {
    case r: MethodRef => !r.methodFullName.startsWith(curFull + ".") && !r.methodFullName.startsWith(curFull + ":")
    case _ => false
  }

  // ---- expression trees ----
  def ex(e: AstNode, d: Int): ujson.Value = {
    if (d > 6) return ujson.Obj("o" -> "unk", "l" -> "depth")
    e match {
      case l: Literal => ujson.Obj("o" -> "lit", "c" -> l.code)
      case i: Identifier => ujson.Obj("o" -> "name", "n" -> i.name)
      case c: Call if opName(c) == "fieldAccess" && isOp(c) =>
        (arg(c, 1), arg(c, 2)) match {
          case (Some(b), Some(f: FieldIdentifier)) => ujson.Obj("o" -> "attr", "b" -> ex(b, d + 1), "f" -> f.canonicalName)
          case _ => ujson.Obj("o" -> "unk", "l" -> "fieldAccess")
        }
      case c: Call if isOp(c) && opName(c) == "formatString" =>
        ujson.Obj("o" -> "fmt", "p" -> ujson.Arr.from(c.argument.l.sortBy(_.argumentIndex).map(a => ex(a, d + 1))))
      case c: Call if isOp(c) && (opName(c) == "addition") =>
        ujson.Obj("o" -> "add", "p" -> ujson.Arr.from(c.argument.l.sortBy(_.argumentIndex).map(a => ex(a, d + 1))))
      case c: Call if !isOp(c) => ujson.Obj("o" -> "call", "n" -> c.name)
      case c: Call => ujson.Obj("o" -> "unk", "l" -> c.name)
      case b: Block => ujson.Obj("o" -> "unk", "l" -> "block")
      case x => ujson.Obj("o" -> "unk", "l" -> x.label)
    }
  }

  // ---- the projection ----
  // Which calls are call sites: non-operator calls, plus `<operator>.new` (TS), minus decorator plumbing.
  def isCallSite(c: Call): Boolean =
    !decoSkip.contains(c.id) && ((!isOp(c) && c.name.nonEmpty) || c.name == "<operator>.new")

  type Def = (AstNode, String, List[String]) // def node (Call | MethodParameterIn), head, fields
  // the projection is intraprocedural (spec §5.10): a def must belong to the method being projected
  var curOwn: Set[Long] = Set.empty
  var curM: Long = -1L

  // DS probe (S28 adjudication, scratch only): relax=true additionally follows a REACHING_DEF whose source is an
  // earlier same-name identifier that is NOT an assignment target (e.g. an earlier call argument) back through that
  // identifier's own reaching defs. rdDrop counts the reaching-def sources the shipped walk drops, by kind.
  var relax = false
  val rdDrop = mutable.Map[String, Int]()
  def bump(k: String): Unit = rdDrop(k) = rdDrop.getOrElse(k, 0) + 1
  def defsOf(e: AstNode, fields: List[String], seen: mutable.Set[(Long, List[String])], depth: Int): List[Def] = {
    if (depth > 40 || !seen.add((e.id, fields))) return Nil
    e match {
      case c: Call if isCallSite(c) => if (curOwn.contains(c.id)) List((c, "r", fields)) else Nil
      case c: Call if isOp(c) && opName(c) == "fieldAccess" =>
        (arg(c, 1), arg(c, 2)) match {
          case (Some(b), Some(f: FieldIdentifier)) => defsOf(b, f.canonicalName :: fields, seen, depth + 1)
          case _ => Nil
        }
      case c: Call if isOp(c) && opName(c) == "indexAccess" =>
        (arg(c, 1), arg(c, 2)) match {
          case (Some(b), Some(k)) =>
            litKey(k) match {
              case Some(key) => defsOf(b, key :: fields, seen, depth + 1)
              case None => defsOf(b, fields, seen, depth + 1)
            }
          case _ => Nil
        }
      case c: Call if isOp(c) && isAssign(c) => arg(c, 2).toList.flatMap(r => defsOf(r, fields, seen, depth + 1))
      case c: Call if isOp(c) =>
        c.argument.l.sortBy(_.argumentIndex).flatMap {
          case _: FieldIdentifier | _: Literal => Nil
          case a => defsOf(a, fields, seen, depth + 1)
        }
      case c: Call => Nil // decorator plumbing / unnamed
      case b: Block =>
        objFields(b) match {
          case Some(fs) =>
            fields match {
              case h :: t => fs.filter(_._1 == h).flatMap { case (_, v) => defsOf(v, t, seen, depth + 1) }
              case Nil => fs.flatMap { case (_, v) => defsOf(v, Nil, seen, depth + 1) }
            }
          case None =>
            kids(b).collect { case x: Expression => x }.lastOption.toList.flatMap(l => defsOf(l, fields, seen, depth + 1))
        }
      case i: Identifier =>
        val rin = i._reachingDefIn.l
        if (!relax && rin.isEmpty) bump("noRdIn")
        rin.flatMap {
          case p: MethodParameterIn => if (p.astParent.id == curM) List((p: AstNode, "p" + p.index, fields)) else Nil
          case s: Identifier if s.id != i.id && s.name == i.name && curOwn.contains(s.id) =>
            s.astParent match {
              case a: Call if isOp(a) && isAssign(a) && arg(a, 1).exists(_.id == s.id) =>
                arg(a, 2).toList.flatMap(r => defsOf(r, fields, seen, depth + 1))
              case par =>
                if (!relax) bump("idNonAssign:" + par.label + ":" + (par match { case c: Call => (if (isOp(c)) c.name else "callsite"); case _ => "" }))
                if (relax) defsOf(s, fields, seen, depth + 1) else Nil
            }
          case x =>
            if (!relax) bump("other:" + x.label)
            Nil
        }
      case _ => Nil
    }
  }

  // use positions of an expression: flatten object literals into fields (cap 3 fields after the head)
  def usePositions(e: Expression, fields: List[String]): List[(List[String], Expression)] = e match {
    case b: Block if fields.size < 3 =>
      objFields(b) match {
        case Some(fs) if fs.nonEmpty => fs.flatMap { case (k, v) => usePositions(v, fields :+ k) }
        case _ => List((fields, e))
      }
    case _ => List((fields, e))
  }

  val histogram = mutable.Map[String, Int]()
  var methodsSeen = 0
  var projNs = 0L
  var cdgRaw = 0
  var cdgSelf = 0
  var dataRaw = 0

  val methods = cpg.method.isExternal(false).l.sortBy(m => (m.filename, line(m), col(m), m.id))
  methods.foreach { m =>
    methodsSeen += 1
    val ps = m.parameter.l.sortBy(_.index)
    emit(ujson.Obj("k" -> "m", "id" -> m.id, "name" -> m.name, "full" -> m.fullName, "file" -> m.filename,
      "l" -> line(m), "le" -> ip(m, "LINE_NUMBER_END"), "c" -> col(m),
      "pk" -> (m._astIn.l.headOption.map(_.label).getOrElse("")), "pid" -> (m._astIn.l.headOption.map(_.id).getOrElse(-1L)), // S28 guard 2: METHOD without an AST parent
      "pfull" -> (m._astIn.l.headOption.map(x => sp(x, "FULL_NAME")).getOrElse("")), "code" -> code(m).take(80)))
    ps.foreach { p =>
      emit(ujson.Obj("k" -> "p", "m" -> m.id, "i" -> p.index, "name" -> p.name, "l" -> line(p), "c" -> col(p), "code" -> code(p).take(80)))
    }

    val body: List[AstNode] = m.astChildren.collectFirst { case b: Block => b }.map(b => ownWalk(b).toList).getOrElse(Nil) // S28 guard 1: bodiless METHOD has no BLOCK

    // decorators (Python): `name = deco(...)(METHOD_REF)` chains in this scope
    body.foreach {
      case a: Call if isOp(a) && isAssign(a) =>
        (arg(a, 1), arg(a, 2)) match {
          case (Some(t: Identifier), Some(rhs)) =>
            decoChain(rhs).foreach { case (chain, ref) =>
              if (lang == "python" && chain.nonEmpty) {
                chain.foreach { c =>
                  decoSkip += c.id
                  if (c.name.isEmpty) kids(c).collectFirst { case k: Call => k }.foreach(k => decoSkip += k.id)
                }
                chain.zipWithIndex.foreach { case (c, ordinal) =>
                  val expr: Option[Call] = if (c.name.isEmpty) kids(c).collectFirst { case k: Call => k } else Some(c)
                  expr.foreach { dc =>
                    val args = dc.argument.l.filter(isPos).sortBy(a => (a.argumentIndex, a.id)).flatMap { a2 =>
                      // a decorator's own receiver is argument 0; positional args are a0.. ; kwargs carry argumentName
                      usePositions(a2, Nil).map { case (fs, ee) =>
                        ujson.Obj("i" -> a2.argumentIndex, "an" -> a2.argumentName.getOrElse(""), "f" -> ujson.Arr.from(fs), "e" -> ex(ee, 0))
                      }
                    }
                    emit(ujson.Obj("k" -> "deco", "m" -> m.id, "target" -> ref.methodFullName, "o" -> ordinal,
                      "name" -> dc.name, "code" -> code(dc).take(160), "l" -> line(dc), "c" -> col(dc),
                      "recv" -> ex(dc.argument.l.find(_.argumentIndex == 0).getOrElse(dc), 0),
                      "args" -> ujson.Arr.from(args)))
                  }
                }
              }
            }
          case _ =>
        }
      case _ =>
    }

    curM = m.id
    curFull = m.fullName
    curOwn = body.map(_.id).toSet
    val callSites = body.collect { case c: Call if isCallSite(c) => c }
    val returns = body.collect { case r: Return => r }
    val csIds = callSites.map(_.id).toSet
    val retIds = returns.map(_.id).toSet

    // conditions = CDG sources among this method's CFG nodes (the CDG decides what counts, spec §5.5)
    val cdgSources = m.cfgNode.l.filter(n => n._cdgOut.nonEmpty && !n.isInstanceOf[Method] && !n.isInstanceOf[MethodReturn])
    val condIds = cdgSources.map(_.id).toSet
    cdgSources.foreach { n =>
      emit(ujson.Obj("k" -> "c", "m" -> m.id, "id" -> n.id, "l" -> line(n), "c" -> col(n), "code" -> code(n).take(400), "lab" -> n.label))
    }

    returns.foreach { r =>
      emit(ujson.Obj("k" -> "r", "m" -> m.id, "id" -> r.id, "l" -> line(r), "c" -> col(r)))
    }

    callSites.foreach { c =>
      val callees = c.callee.l
      val inRepo = callees.filter(x => !x.isExternal)
      val outRepo = callees.filter(_.isExternal).map(_.fullName).distinct.sorted
      val args = c.argument.l.filter(a => isArg(a) && !isArtifactRef(a)).sortBy(a => (a.argumentIndex, a.id)).flatMap { a =>
        usePositions(a, Nil).map { case (fs, ee) =>
          ujson.Obj("i" -> a.argumentIndex, "an" -> a.argumentName.getOrElse(""), "f" -> ujson.Arr.from(fs), "e" -> ex(ee, 0))
        }
      }
      emit(ujson.Obj("k" -> "cs", "m" -> m.id, "id" -> c.id, "name" -> c.name, "mfn" -> c.methodFullName,
        "l" -> line(c), "c" -> col(c), "code" -> code(c).take(2000), "dt" -> c.dispatchType,
        "in" -> ujson.Arr.from(inRepo.sortBy(_.id).map(x => ujson.Obj("id" -> x.id, "full" -> x.fullName))),
        "out" -> ujson.Arr.from(outRepo), "args" -> ujson.Arr.from(args)))
    }

    // module / type level constants: direct block children `NAME = <rhs>`
    if (m.name == "<module>" || m.name == ":program" || m.name == "<body>") {
      val tops = kids(m.block).collect { case a: Call if isOp(a) && isAssign(a) => a }
      tops.foreach { a =>
        (arg(a, 1), arg(a, 2)) match {
          case (Some(t: Identifier), Some(rhs)) if decoChain(rhs).isEmpty && !t.name.startsWith("tmp") && !t.name.startsWith("_tmp_") =>
            emit(ujson.Obj("k" -> "const", "m" -> m.id, "name" -> t.name, "l" -> line(a), "e" -> ex(rhs, 0)))
          case _ =>
        }
      }
    }

    // ---- projection: data ----
    val t0 = System.nanoTime
    def emitData(useKind: String, useId: Long, toHead: String, toFields: List[String], expr: AstNode): Unit = {
      val seen = mutable.Set[(Long, List[String])]()
      relax = true
      val seen2 = mutable.Set[(Long, List[String])]()
      defsOf(expr, Nil, seen2, 0).foreach { case (dn, head, fs) =>
        val defObj2 = dn match {
          case c: Call => ujson.Obj("dk" -> "cs", "did" -> c.id)
          case p: MethodParameterIn => ujson.Obj("dk" -> "p", "did" -> m.id, "di" -> p.index)
          case _ => ujson.Obj("dk" -> "?", "did" -> dn.id)
        }
        val o2 = ujson.Obj("k" -> "d2", "m" -> m.id, "uk" -> useKind, "uid" -> useId, "th" -> toHead,
          "tf" -> ujson.Arr.from(toFields), "h" -> head, "ff" -> ujson.Arr.from(fs))
        defObj2.obj.foreach { case (k, v) => o2(k) = v }
        emit(o2)
      }
      relax = false
      defsOf(expr, Nil, seen, 0).foreach { case (dn, head, fs) =>
        dataRaw += 1
        val defObj = dn match {
          case c: Call => ujson.Obj("dk" -> "cs", "did" -> c.id)
          case p: MethodParameterIn => ujson.Obj("dk" -> "p", "did" -> m.id, "di" -> p.index)
          case _ => ujson.Obj("dk" -> "?", "did" -> dn.id)
        }
        val o = ujson.Obj("k" -> "d", "m" -> m.id, "uk" -> useKind, "uid" -> useId, "th" -> toHead,
          "tf" -> ujson.Arr.from(toFields), "h" -> head, "ff" -> ujson.Arr.from(fs))
        defObj.obj.foreach { case (k, v) => o(k) = v }
        emit(o)
      }
    }
    callSites.foreach { c =>
      c.argument.l.filter(isArg).sortBy(a => (a.argumentIndex, a.id)).foreach { a =>
        val head = if (a.argumentName.exists(_.nonEmpty)) "kw." + a.argumentName.get else if (a.argumentIndex == 0) "recv" else "a" + (a.argumentIndex - 1)
        usePositions(a, Nil).foreach { case (fs, ee) => emitData("cs", c.id, head, fs, ee) }
      }
    }
    returns.foreach { r =>
      kids(r).collect { case e: Expression => e }.headOption.foreach { e =>
        usePositions(e, Nil).foreach { case (fs, ee) => emitData("r", r.id, "ret", fs, ee) }
      }
    }
    cdgSources.foreach { n => emitData("c", n.id, "", Nil, n) }

    // ---- projection: controlledBy (CDG, raw) ----
    def pointOf(t: AstNode): Option[(String, Long)] = {
      var cur: AstNode = t
      while (cur != null) {
        cur match {
          case x if condIds.contains(x.id) => return Some(("c", x.id))
          case c: Call if csIds.contains(c.id) => return Some(("cs", c.id))
          case r: Return if retIds.contains(r.id) => return Some(("r", r.id))
          case _: Method => return None
          case _ =>
        }
        cur = cur.astParent
      }
      None
    }
    cdgSources.foreach { k =>
      k._cdgOut.l.foreach { tgt =>
        tgt match {
          case ta: AstNode =>
            pointOf(ta).foreach { case (pk, pid) =>
              cdgRaw += 1
              if (pk == "c" && pid == k.id) cdgSelf += 1
              emit(ujson.Obj("k" -> "cb", "m" -> m.id, "pk" -> pk, "pid" -> pid, "cond" -> k.id))
            }
          case _ =>
        }
      }
    }
    val dt = System.nanoTime - t0
    projNs += dt
    val bucket = if (dt < 1000000L) "<1ms" else if (dt < 10000000L) "1-10ms" else if (dt < 100000000L) "10-100ms" else if (dt < 1000000000L) "100ms-1s" else ">=1s"
    histogram(bucket) = histogram.getOrElse(bucket, 0) + 1
  }

  // type declarations (structure only; normalize.py decides which are symbols)
  cpg.typeDecl.isExternal(false).l.sortBy(t => (t.filename, line(t), col(t), t.id)).foreach { t =>
    emit(ujson.Obj("k" -> "t", "id" -> t.id, "name" -> t.name, "full" -> t.fullName, "file" -> t.filename,
      "l" -> line(t), "c" -> col(t), "pk" -> (t._astIn.l.headOption.map(_.label).getOrElse("")), "pid" -> (t._astIn.l.headOption.map(_.id).getOrElse(-1L)), "pfull" -> (t._astIn.l.headOption.map(x => sp(x, "FULL_NAME")).getOrElse(""))))
  }
  // files the frontend parsed (the per-file rule's denominator is measured by cpg-content-check.sc)
  cpg.file.l.sortBy(_.name).foreach { f => emit(ujson.Obj("k" -> "f", "name" -> f.name)) }

  val tEnd = System.nanoTime
  emit(ujson.Obj("k" -> "stats", "loadMs" -> ((tLoaded - tStart) / 1000000L), "projectionMs" -> (projNs / 1000000L),
    "totalMs" -> ((tEnd - tStart) / 1000000L), "methods" -> methodsSeen, "cdgRaw" -> cdgRaw, "cdgSelf" -> cdgSelf,
    "dataRaw" -> dataRaw, "rdDrop" -> ujson.Obj.from(rdDrop.toList.sortBy(_._1).map { case (k, v) => k -> ujson.Num(v) }), "histogram" -> ujson.Obj.from(histogram.toList.sortBy(_._1).map { case (k, v) => k -> ujson.Num(v) })))
  w.close()
  println(s"RDRELAX lang=$lang methods=$methodsSeen cdgRaw=$cdgRaw cdgSelf=$cdgSelf dataRaw=$dataRaw projectionMs=${projNs / 1000000L}")
}
