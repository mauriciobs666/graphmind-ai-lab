const val TOP = "top"
@Foo("fa")
fun a(): Int { return g("s", "a" + "b", TOP) }
fun b(TOP: String) { g(TOP) }
abstract class K(val p: Int) {
    abstract fun ab()
    external fun ex()
}
expect fun e()
