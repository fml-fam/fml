#include "../catch.hpp"

#include <_internals/arraytools/src/arraytools.hpp>
#include <mpi/dimops.hh>

using namespace arraytools;

extern grid g;


TEMPLATE_TEST_CASE("dimops - sums", "[dimops]", float, double)
{
  len_t m = 3;
  len_t n = 2;
  
  mpimat<TestType> x(g, m, n, 1, 1);
  x.fill_linspace(1, m*n);
  
  cpuvec<TestType> s;
  dimops::rowsums(x, s);
  REQUIRE( s.size() == 3 );
  REQUIRE( fltcmp::eq(s.get(0), 5) );
  REQUIRE( fltcmp::eq(s.get(1), 7) );
  REQUIRE( fltcmp::eq(s.get(2), 9) );
  
  dimops::colsums(x, s);
  REQUIRE( s.size() == 2 );
  REQUIRE( fltcmp::eq(s.get(0), 6) );
  REQUIRE( fltcmp::eq(s.get(1), 15) );
}



TEMPLATE_TEST_CASE("dimops - means", "[dimops]", float, double)
{
  len_t m = 3;
  len_t n = 2;
  
  mpimat<TestType> x(g, m, n, 1, 1);
  x.fill_linspace(1, m*n);
  
  cpuvec<TestType> s;
  dimops::rowmeans(x, s);
  REQUIRE( s.size() == 3 );
  REQUIRE( fltcmp::eq(s.get(0), 2.5) );
  REQUIRE( fltcmp::eq(s.get(1), 3.5) );
  REQUIRE( fltcmp::eq(s.get(2), 4.5) );
  
  dimops::colmeans(x, s);
  REQUIRE( s.size() == 2 );
  REQUIRE( fltcmp::eq(s.get(0), 2) );
  REQUIRE( fltcmp::eq(s.get(1), 5) );
}



TEMPLATE_TEST_CASE("dimops - scale", "[dimops]", float, double)
{
  len_t m = 3;
  len_t n = 2;
  
  mpimat<TestType> x(g, m, n, 1, 1);
  x.fill_linspace(1, m*n);
  
  dimops::scale(true, true, x);
  
  REQUIRE( fltcmp::eq(x.get(0, 0), -1) );
  REQUIRE( fltcmp::eq(x.get(0, 1), -1) );
  
  REQUIRE( fltcmp::eq(x.get(1, 0), 0) );
  REQUIRE( fltcmp::eq(x.get(1, 1), 0) );
  
  REQUIRE( fltcmp::eq(x.get(2, 0), 1) );
  REQUIRE( fltcmp::eq(x.get(2, 1), 1) );
}