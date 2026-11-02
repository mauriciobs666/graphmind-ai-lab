// Smoke project: one interface, one implementation, one call through the interface.
export interface Shape {
  area(): number;
}

export class Circle implements Shape {
  constructor(private r: number) {}
  area(): number {
    return 3 * this.r * this.r;
  }
}

export function total(shape: Shape): number {
  return shape.area();
}

export function main(): number {
  return total(new Circle(2));
}
