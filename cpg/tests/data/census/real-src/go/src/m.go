package p
const K = "k"
type I interface { A() }
type J = interface { B() }
func f(x interface{ Close() error }) {}
func asm(a int) int
func g() { call("s", "a" + "b", `raw`, K) }
func sh(K string) { call(K) }
