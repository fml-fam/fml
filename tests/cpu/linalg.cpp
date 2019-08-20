#include "../catch.hpp"

#include <cpu/cpumat.hh>
#include <cpu/linalg.hh>


static inline bool test_eq(const float x, const float y)
{
  const float eps = 1e-5;
  return (std::abs(x-y) < eps);
}



TEST_CASE("matrix multiplication", "[linalg]")
{
  len_t n = 2;
  
  cpumat<float> x = cpumat<float>(n, n);
  cpumat<float> y = cpumat<float>(n, n);
  
  float *x_d = x.data_ptr();
  float *y_d = y.data_ptr();
  
  for (len_t i=0; i<n*n; i++)
    x_d[i] = (float) i+1;
  
  for (len_t i=0; i<n*n; i++)
    y_d[i] = (float) (n*n)-i;
  
  cpumat<float> z = linalg::matmult(false, false, 1.0f, x, y);
  const float *data = z.data_ptr();
  REQUIRE( (
    test_eq(data[0], 13.f) &&
    test_eq(data[1], 20.f) &&
    test_eq(data[2], 5.f) &&
    test_eq(data[3], 8.f)
  ) );
  
  linalg::matmult_noalloc(true, false, 1.0f, x, y, z);
  REQUIRE( (
    test_eq(data[0], 10.f) &&
    test_eq(data[1], 24.f) &&
    test_eq(data[2], 4.f) &&
    test_eq(data[3], 10.f)
  ) );
  
  linalg::matmult_noalloc(false, true, 1.0f, x, y, z);
  REQUIRE( (
    test_eq(data[0], 10.f) &&
    test_eq(data[1], 16.f) &&
    test_eq(data[2], 6.f) &&
    test_eq(data[3], 10.f)
  ) );
  
  linalg::matmult_noalloc(true, true, 1.0f, x, y, z);
  REQUIRE( (
    test_eq(data[0], 8.f) &&
    test_eq(data[1], 20.f) &&
    test_eq(data[2], 5.f) &&
    test_eq(data[3], 13.f)
  ) );
}
 
