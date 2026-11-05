class R {
  fun run() { obj.call("x", "y"); this.call("w") }
  @GetMapping("/a") @Deprecated("old") fun m2() {}
}
