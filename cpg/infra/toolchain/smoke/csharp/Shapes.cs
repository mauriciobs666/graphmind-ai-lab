// Smoke project: one interface, one implementation, one call through the interface.
public interface IShape
{
    double Area();
}

public class Circle : IShape
{
    private readonly double r;

    public Circle(double r) { this.r = r; }

    public double Area() => 3.0 * r * r;
}

public static class Program
{
    public static double Total(IShape shape) => shape.Area();

    public static void Main() => System.Console.WriteLine(Total(new Circle(2.0)));
}
