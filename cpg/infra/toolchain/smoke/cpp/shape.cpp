#include "shape.h"

double Circle::area() const { return 3.0 * r_ * r_; }

double total(const std::vector<const Shape*>& shapes, std::atomic<std::size_t>& calls) {
  double sum = 0.0;
  for (const Shape* s : shapes) {
    sum += s->area();
    calls.fetch_add(1);
  }
  return sum;
}
