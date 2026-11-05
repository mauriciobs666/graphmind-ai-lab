class Q {
  run() { obj.call("x", "y"); this.call("w"); }
}
@Component({selector: "s"})
class Cmp { @Input() x: string; @HostListener("click", ["$event"]) onClick(e) {} }
function plain() {}
