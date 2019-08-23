#include "../catch.hpp"
#include "../fltcmp.hh"

#include <cpu/cpumat.hh>
#include <cpu/linalg.hh>



TEMPLATE_TEST_CASE("matrix multiplication", "[linalg]", float, double)
{
  len_t n = 2;
  
  cpumat<TestType> x(n, n);
  cpumat<TestType> y(n, n);
  
  x.fill_linspace(1.f, (float) n*n);
  y.fill_linspace((float) n*n, 1.f);
  
  cpumat<TestType> z = linalg::matmult(false, false, (TestType)1, x, y);
  const TestType *data = z.data_ptr();
  REQUIRE( fltcmp::eq(data[0], 13) );
  REQUIRE( fltcmp::eq(data[1], 20) );
  REQUIRE( fltcmp::eq(data[2], 5) );
  REQUIRE( fltcmp::eq(data[3], 8) );
  
  linalg::matmult_noalloc(true, false, (TestType)1, x, y, z);
  REQUIRE( fltcmp::eq(data[0], 10) );
  REQUIRE( fltcmp::eq(data[1], 24) );
  REQUIRE( fltcmp::eq(data[2], 4) );
  REQUIRE( fltcmp::eq(data[3], 10) );
  
  linalg::matmult_noalloc(false, true, (TestType)1, x, y, z);
  REQUIRE( fltcmp::eq(data[0], 10) );
  REQUIRE( fltcmp::eq(data[1], 16) );
  REQUIRE( fltcmp::eq(data[2], 6) );
  REQUIRE( fltcmp::eq(data[3], 10) );
  
  linalg::matmult_noalloc(true, true, (TestType)1, x, y, z);
  REQUIRE( fltcmp::eq(data[0], 8) );
  REQUIRE( fltcmp::eq(data[1], 20) );
  REQUIRE( fltcmp::eq(data[2], 5) );
  REQUIRE( fltcmp::eq(data[3], 13) );
}
