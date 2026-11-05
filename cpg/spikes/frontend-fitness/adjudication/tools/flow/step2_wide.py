s=open('rdrelax.sc').read()
def rep(old,new):
    global s
    assert old in s, old[:70]; s=s.replace(old,new)
rep("  var relax = false\n", "  var relax = false\n  var wide = false\n")
rep("""                if (relax) defsOf(s, fields, seen, depth + 1) else Nil
            }
          case x =>
            if (!relax) bump("other:" + x.label)
            Nil""",
"""                if (relax) defsOf(s, fields, seen, depth + 1) else Nil
            }
          case s: Identifier if wide && s.id != i.id && curOwn.contains(s.id) => defsOf(s, fields, seen, depth + 1)
          case c: Call if wide && curOwn.contains(c.id) => defsOf(c, fields, seen, depth + 1)
          case x =>
            if (!relax) bump("other:" + x.label)
            Nil""")
# emit d3 using wide=true (relax also true)
rep("""      relax = false
      defsOf(expr, Nil, seen, 0).foreach { case (dn, head, fs) =>""",
"""      wide = true
      val seen3 = mutable.Set[(Long, List[String])]()
      defsOf(expr, Nil, seen3, 0).foreach { case (dn, head, fs) =>
        val defObj3 = dn match {
          case c: Call => ujson.Obj("dk" -> "cs", "did" -> c.id)
          case p: MethodParameterIn => ujson.Obj("dk" -> "p", "did" -> m.id, "di" -> p.index)
          case _ => ujson.Obj("dk" -> "?", "did" -> dn.id)
        }
        val o3 = ujson.Obj("k" -> "d3", "m" -> m.id, "uk" -> useKind, "uid" -> useId, "th" -> toHead,
          "tf" -> ujson.Arr.from(toFields), "h" -> head, "ff" -> ujson.Arr.from(fs))
        defObj3.obj.foreach { case (k, v) => o3(k) = v }
        emit(o3)
      }
      wide = false
      relax = false
      defsOf(expr, Nil, seen, 0).foreach { case (dn, head, fs) =>""")
s=s.replace('RDRELAX lang','RDWIDE lang')
open('rdwide.sc','w').write(s)
