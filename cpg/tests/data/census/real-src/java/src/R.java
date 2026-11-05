public class R {
  void run() { obj.call("x", "y"); this.call("w"); }
  @RequestMapping(value = "/a", method = RequestMethod.GET) @Deprecated void m2() {}
  @SuppressWarnings({"a", "b"}) void m3() {}
}
