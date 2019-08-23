#include "../catch.hpp"
#include "../fltcmp.hh"

#include <mpi/grid.hh>
#include <mpi/linalg.hh>
#include <mpi/mpimat.hh>
#include <mpi/mpihelpers.hh>

extern grid g;


TEMPLATE_TEST_CASE("matrix multiplication", "[linalg]", float, double)
{
  len_t n = 2;
  
  mpimat<TestType> x(g, n, n, 1, 1);
  mpimat<TestType> y(g, n, n, 1, 1);
  x.fill_linspace(1.f, (TestType) n*n);
  y.fill_linspace((TestType) n*n, 1.f);
  
  mpimat<TestType> z = linalg::matmult(false, false, (TestType)1, x, y);
  REQUIRE( fltcmp::eq(z(0), 13) );
  REQUIRE( fltcmp::eq(z(1), 20) );
  REQUIRE( fltcmp::eq(z(2), 5) );
  REQUIRE( fltcmp::eq(z(3), 8) );
  
  linalg::matmult_noalloc(true, false, (TestType)1, x, y, z);
  REQUIRE( fltcmp::eq(z(0), 10) );
  REQUIRE( fltcmp::eq(z(1), 24) );
  REQUIRE( fltcmp::eq(z(2), 4) );
  REQUIRE( fltcmp::eq(z(3), 10) );
  
  linalg::matmult_noalloc(false, true, (TestType)1, x, y, z);
  REQUIRE( fltcmp::eq(z(0), 10) );
  REQUIRE( fltcmp::eq(z(1), 16) );
  REQUIRE( fltcmp::eq(z(2), 6) );
  REQUIRE( fltcmp::eq(z(3), 10) );
  
  linalg::matmult_noalloc(true, true, (TestType)1, x, y, z);
  REQUIRE( fltcmp::eq(z(0), 8) );
  REQUIRE( fltcmp::eq(z(1), 20) );
  REQUIRE( fltcmp::eq(z(2), 5) );
  REQUIRE( fltcmp::eq(z(3), 13) );
}
 
