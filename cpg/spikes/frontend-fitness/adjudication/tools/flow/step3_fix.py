s=open('rdrelax.sc').read()
def rep(old,new):
    global s
    assert old in s, old[:70]; s=s.replace(old,new)
rep("  var relax = false\n", "  var relax = false\n  var ctor = false\n")
rep("""              case par =>
                if (!relax) bump(""",
"""              case c: Call if ctor && c.name == "<init>" && s.argumentIndex == 0 && curOwn.contains(c.id) => List((c: AstNode, "r", fields))
              case par =>
                if (!relax) bump(""")
# d2 block now runs with ctor=true too -> rename to d4 semantics: relaxed+ctor
rep("""      relax = true
      val seen2""", """      relax = true
      ctor = true
      val seen2""")
rep("""      relax = false
      defsOf(expr, Nil, seen, 0)""", """      relax = false
      ctor = false
      defsOf(expr, Nil, seen, 0)""")
s=s.replace('"k" -> "d2"','"k" -> "d4"').replace('RDRELAX lang','RDFIX lang')
open('rdfix.sc','w').write(s)
