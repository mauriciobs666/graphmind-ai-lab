// astgen.sc -- P1-04 port of the S11 helper (body unchanged): run the jssrc2cpg bundled `astgen` over the staged unit and leave the Babel
// ASTs in the output dir, for normalize.py (JS/TS imports and type-declaration end lines, which the
// CPG does not carry) and verify_qnames.py (an independent qname derivation). Run only through the
// runner's `joern --script` form:
//   offline-run.sh ... --frontend joern --script-file astgen.sc --source <unit> --output <dir> \
//     -- --script /scripts/astgen.sc --param cpgFile=unused --param outDir=/out/@AUDIT_ID@.astgen
// astgen is the same binary jssrc2cpg runs; it reads /src and writes JSON under outDir.
import scala.sys.process._
@main def main(cpgFile: String, outDir: String) = {
  val astgen = "/opt/joern/joern-cli/frontends/jssrc2cpg/bin/astgen/astgen-linux"
  val sb = new StringBuilder
  val rc = Process(Seq(astgen, "-t", "ts", "-o", outDir, "-r"), new java.io.File("/src"))
    .!(ProcessLogger(l => sb.append(l).append("\n"), l => sb.append(l).append("\n")))
  println(s"ASTGEN rc=$rc files=" + sb.toString.split("\n").count(_.startsWith("Converted AST for")))
  if (rc != 0) { System.out.flush(); sys.exit(rc) }
}
