#include <immintrin.h>  // compiler-builtin header: comes from the compiler's resource dir
#include <cstdio>

#include "shape.h"

int main() {
  Circle c(2.0);
  std::vector<const Shape*> shapes{&c};
  std::atomic<std::size_t> calls{0};
  __m128 v = _mm_add_ps(_mm_set1_ps(1.0f), _mm_set1_ps(2.0f));
  std::printf("%f %f\n", total(shapes, calls), _mm_cvtss_f32(v));
  return 0;
}
