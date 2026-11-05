struct R { void m(); };
void run(R* p, R o) { p->call("x", "y"); o.call("w"); }
[[deprecated("old")]] void d1() {}
[[nodiscard]] int d2() { return 0; }
