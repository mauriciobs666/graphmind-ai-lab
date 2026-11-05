p='rdrelax.sc'; s=open(p).read()
def rep(old,new):
    global s
    assert old in s, old[:60]; s=s.replace(old,new)
rep("""  def defsOf(e: AstNode, fields: List[String], seen: mutable.Set[(Long, List[String])], depth: Int): List[Def] = {
    if (depth > 40 || !seen.add((e.id, fields))) return Nil""",
"""  // DS probe (S28 adjudication, scratch only): relax=true additionally follows a REACHING_DEF whose source is an
  // earlier same-name identifier that is NOT an assignment target (e.g. an earlier call argument) back through that
  // identifier's own reaching defs. rdDrop counts the reaching-def sources the shipped walk drops, by kind.
  var relax = false
  val rdDrop = mutable.Map[String, Int]()
  def bump(k: String): Unit = rdDrop(k) = rdDrop.getOrElse(k, 0) + 1
  def defsOf(e: AstNode, fields: List[String], seen: mutable.Set[(Long, List[String])], depth: Int): List[Def] = {
    if (depth > 40 || !seen.add((e.id, fields))) return Nil""")
rep("""      case i: Identifier =>
        i._reachingDefIn.l.flatMap {
          case p: MethodParameterIn => if (p.astParent.id == curM) List((p: AstNode, "p" + p.index, fields)) else Nil
          case s: Identifier if s.id != i.id && s.name == i.name && curOwn.contains(s.id) =>
            s.astParent match {
              case a: Call if isOp(a) && isAssign(a) && arg(a, 1).exists(_.id == s.id) =>
                arg(a, 2).toList.flatMap(r => defsOf(r, fields, seen, depth + 1))
              case _ => Nil
            }
          case _ => Nil
        }""",
"""      case i: Identifier =>
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
        }""")
rep("""      defsOf(expr, Nil, seen, 0).foreach { case (dn, head, fs) =>""",
"""      relax = true
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
      defsOf(expr, Nil, seen, 0).foreach { case (dn, head, fs) =>""")
rep('''"dataRaw" -> dataRaw, "histogram"''',
'''"dataRaw" -> dataRaw, "rdDrop" -> ujson.Obj.from(rdDrop.toList.sortBy(_._1).map { case (k, v) => k -> ujson.Num(v) }), "histogram"''')
s=s.replace('println(s"EXTRACT lang','println(s"RDRELAX lang')
open(p,'w').write(s)
