class R {
  void Run() { obj.Call("x", "y"); this.Call("w"); }
  [Route("/a"), Obsolete("old")] void M2() {}
  [HttpGet] void M3() {}
}
