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
  cpumat<TestType> x_cpu = cpumat<TestType>(n, n);
  cpumat<TestType> y_cpu = cpumat<TestType>(n, n);
  
  TestType *x_d = x_cpu.data_ptr();
  TestType *y_d = y_cpu.data_ptr();
  
  for (len_t i=0; i<n*n; i++)
  {
    x_d[i] = (TestType) i+1;
    y_d[i] = (TestType) (n*n)-i;
  }
  
  mpimat<TestType> x = mpihelpers::cpu2mpi(x_cpu, g, 1, 1);
  mpimat<TestType> y = mpihelpers::cpu2mpi(y_cpu, g, 1, 1);
  
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
 
