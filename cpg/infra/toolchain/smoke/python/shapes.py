"""Smoke project: one base class, one override, one call through the base."""


class Shape:
    def area(self) -> float:
        raise NotImplementedError


class Circle(Shape):
    def __init__(self, r: float) -> None:
        self.r = r

    def area(self) -> float:
        return 3.0 * self.r * self.r


def total(shape: Shape) -> float:
    return shape.area()


def main() -> float:
    return total(Circle(2.0))
