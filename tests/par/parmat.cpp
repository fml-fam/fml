#include "../catch.hpp"

#include <arraytools/src/arraytools.hpp>
#include <par/cpu.hh>

using namespace arraytools;

extern comm r;


TEMPLATE_TEST_CASE("basics", "[parmat_cpu]", float, double)
{
  len_global_t m = 3;
  len_t n = 2;
  
  parmat_cpu<TestType> x(r, m, n);
  REQUIRE( x.nrows() == m );
  REQUIRE( x.ncols() == n );
  
  // x.fill_zero();
  // REQUIRE( fltcmp::eq(x(0, 0), 0) );
  // if (m.rank0())
  //   x.data_ptr()[0] = (TestType) 3.14;
  // 
  // REQUIRE( fltcmp::eq(x(0, 0), 3.14) );
}



TEMPLATE_TEST_CASE("scale", "[parmat_cpu]", float, double)
{
  len_t m = 6;
  len_t n = 5;

  parmat_cpu<TestType> x(r, m, n);
  x.fill_one();

  x.scale(3.0f);
  // REQUIRE( fltcmp::eq(x(0), 3) );
  // REQUIRE( fltcmp::eq(x(1), 3) );
}



// TEMPLATE_TEST_CASE("indexing", "[parmat_cpu]", float, double)
// {
//   len_t n = 2;
// 
//   cpumat<TestType> x_cpu = cpumat<TestType>(n, n);
// 
//   TestType *x_d = x_cpu.data_ptr();
// 
//   for (len_t i=0; i<n*n; i++)
//     x_d[i] = (TestType) i+1;
// 
//   mpimat<TestType> x = mpihelpers::cpu2mpi(x_cpu, g, 1, 1);
//   mpimat<TestType> y(g, n, n, 1, 1);
// 
//   y.fill_linspace(1, n*n);
//   REQUIRE( (x == y) );
// 
//   y.fill_val(1.f);
//   REQUIRE( (x != y) );
// }



TEMPLATE_TEST_CASE("diag", "[cpumat]", float, double)
{
  len_t m = 4;
  len_t n = 3;
  
  parmat_cpu<TestType> x(r, m, n);
  x.fill_linspace(1, m*n);
  
  // cpuvec<TestType> v;
  // x.diag(v);
  // REQUIRE( fltcmp::eq(v(0), 1) );
  // REQUIRE( fltcmp::eq(v(1), 6) );
  // REQUIRE( fltcmp::eq(v(2), 11) );
  // 
  // x.antidiag(v);
  // REQUIRE( fltcmp::eq(v(0), 4) );
  // REQUIRE( fltcmp::eq(v(1), 7) );
  // REQUIRE( fltcmp::eq(v(2), 10) );
}
