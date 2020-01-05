#include "../catch.hpp"

#include <_internals/arraytools/src/arraytools.hpp>
#include <par/cpu.hh>

using namespace arraytools;

extern comm r;


TEMPLATE_TEST_CASE("crossprod", "[linalg]", float, double)
{
  len_global_t m = 3;
  len_t n = 2;
  
  parmat_cpu<TestType> x(r, m, n);
  x.fill_linspace(1, m*n);
  
  // regular api
  cpumat<TestType> x_cp = linalg::crossprod((TestType)1, x);
  REQUIRE( x_cp.nrows() == x.ncols() );
  REQUIRE( x_cp.ncols() == x.ncols() );
  
  REQUIRE( fltcmp::eq(x_cp.get(0, 0), 14) );
  REQUIRE( fltcmp::eq(x_cp.get(1, 0), 32) );
  REQUIRE( fltcmp::eq(x_cp.get(1, 1), 77) );
  
  // noalloc api
  x.fill_linspace(m*n, 1);
  
  linalg::crossprod((TestType)1, x, x_cp);
  REQUIRE( fltcmp::eq(x_cp.get(0, 0), 77) );
  REQUIRE( fltcmp::eq(x_cp.get(1, 0), 32) );
  REQUIRE( fltcmp::eq(x_cp.get(1, 1), 14) );
}
