public class Main {
  static double total(Shape shape) {
    return shape.area();
  }

  public static void main(String[] args) {
    System.out.println(total(new Circle(2.0)));
  }
}
