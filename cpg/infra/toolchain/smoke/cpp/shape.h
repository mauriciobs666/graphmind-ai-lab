#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

struct Shape {
  virtual ~Shape() = default;
  virtual double area() const = 0;
};

struct Circle : Shape {
  explicit Circle(double r) : r_(r) {}
  double area() const override;

 private:
  double r_;
};

// Call through the interface, over a std::vector and a std::atomic: the TU only parses if
// the standard-library and compiler-builtin headers resolve.
double total(const std::vector<const Shape*>& shapes, std::atomic<std::size_t>& calls);
