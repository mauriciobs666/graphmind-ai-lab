const char* const K = "k";
class A {
 public:
  A() = default;
  A(const A&) = delete;
  virtual void pv() = 0;
  void decl(int) const;
  friend void fr(A&);
};
[[nodiscard("why")]] int f() { call("s", "a" "b", R"(raw)", K); return 0; }
int proto(int);
void sh(int K) { call(K); }
#define MAC(x) x
