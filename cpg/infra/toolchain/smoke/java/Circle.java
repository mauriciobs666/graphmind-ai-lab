public class Circle implements Shape {
  private final double r;

  public Circle(double r) {
    this.r = r;
  }

  @Override
  public double area() {
    return 3.0 * r * r;
  }
}
