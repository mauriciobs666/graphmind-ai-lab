// Smoke project: one interface, one implementation, one call through the interface.
package main

import "fmt"

type Shape interface {
	Area() float64
}

type Circle struct{ r float64 }

func (c Circle) Area() float64 { return 3.0 * c.r * c.r }

func total(s Shape) float64 { return s.Area() }

func main() {
	fmt.Println(total(Circle{2.0}))
}
