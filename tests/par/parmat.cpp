#include "../catch.hpp"

#include <fml/_internals/arraytools/src/arraytools.hpp>
#include <fml/par/cpu.hh>

using namespace arraytools;

extern fml::comm r;


TEMPLATE_TEST_CASE("basics", "[parmat_cpu]", float, double)
{
  len_global_t m = 3;
  len_t n = 2;
  
  fml::parmat_cpu<TestType> x(r, m, n);
  REQUIRE( x.nrows() == m );
  REQUIRE( x.ncols() == n );
  
  x.fill_zero();
  REQUIRE( fltcmp::eq(x.get(0, 0), 0) );
  x.set(0, 0, 3.14);
  REQUIRE( fltcmp::eq(x.get(0, 0), 3.14) );
}



TEMPLATE_TEST_CASE("scale", "[parmat_cpu]", float, double)
{
  len_t m = 3;
  len_t n = 2;
  
  fml::parmat_cpu<TestType> x(r, m, n);
  x.fill_val(1);
  x.scale((TestType) 3);
  
  REQUIRE( fltcmp::eq(x.get(0, 0), 3) );
  REQUIRE( fltcmp::eq(x.get(2, 0), 3) );
}



TEMPLATE_TEST_CASE("indexing", "[parmat_cpu]", float, double)
{
  len_t n = 4;
  
  fml::parmat_cpu<TestType> x(r, n, n);
  fml::parmat_cpu<TestType> y(r, n, n);
  
  for (len_t i=0; i<n*n; i++)
    x.set(i, i+1);
  
  y.fill_linspace(1, n*n);
  REQUIRE( (x == y) );
  
  y.fill_val(1.f);
  REQUIRE( (x != y) );
}



TEMPLATE_TEST_CASE("rev", "[parmat_cpu]", float, double)
{
  len_t n = 2;
  
  fml::parmat_cpu<TestType> x(r, n, n);
  x.fill_linspace(1, n*n);
  
  x.rev_cols();
  REQUIRE( fltcmp::eq(x.get(0, 0), 3) );
  REQUIRE( fltcmp::eq(x.get(1, 0), 4) );
  
  // x.rev_rows();
  // REQUIRE( fltcmp::eq(x.get(0, 0), 4) );
  // REQUIRE( fltcmp::eq(x.get(1, 0), 3) );
}
