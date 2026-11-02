// Smoke project: one interface, one override, one call through the interface.
interface Shape {
    fun area(): Double
}

class Circle(private val r: Double) : Shape {
    override fun area(): Double = 3.0 * r * r
}

fun total(shape: Shape): Double = shape.area()

fun main() {
    println(total(Circle(2.0)))
}
