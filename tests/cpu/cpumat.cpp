#include "../catch.hpp"

#include <fltcmp.hh>
#include <cpu/cpumat.hh>



TEMPLATE_TEST_CASE("basics", "[cpumat]", float, double)
{
  len_t m = 3;
  len_t n = 2;
  
  cpumat<TestType> x(m, n);
  REQUIRE( x.nrows() == m );
  REQUIRE( x.ncols() == n );
  
  x.fill_zero();
  REQUIRE( fltcmp::eq(x(0, 0), 0) );
  x(0, 0) = (TestType) 3.14;
  REQUIRE( fltcmp::eq(x(0, 0), 3.14) );
}



TEMPLATE_TEST_CASE("inheriting memory", "[cpumat]", float, double)
{
  len_t m = 2;
  len_t n = 3;
  
  TestType *data = (TestType*) malloc(m*n*sizeof(*data));
  cpumat<TestType> x(data, m, n);
  x.fill_eye();
  x.~cpumat();
  REQUIRE( fltcmp::eq(data[0], 1) );
  
  cpumat<TestType> y;
  y.set(data, m, n);
  y.fill_zero();
  y.~cpumat();
  REQUIRE( fltcmp::eq(data[0], 0) );
  
  free(data);
}
 


TEMPLATE_TEST_CASE("resize", "[cpumat]", float, double)
{
  len_t m = 3;
  len_t n = 2;
  
  cpumat<TestType> x;
  x.resize(m, n);
  x.fill_eye();
  
  REQUIRE( x.nrows() == m );
  REQUIRE( x.ncols() == n );
  
  REQUIRE( fltcmp::eq(x(0), 1) );
  REQUIRE( fltcmp::eq(x(1), 0) );
}



TEMPLATE_TEST_CASE("scale", "[cpumat]", float, double)
{
  len_t m = 3;
  len_t n = 2;
  
  cpumat<TestType> x(m, n);
  x.fill_one();
  
  x.scale((TestType) 3);
  REQUIRE( fltcmp::eq(x(0), 3) );
  REQUIRE( fltcmp::eq(x(1), 3) );
}



TEMPLATE_TEST_CASE("indexing", "[cpumat]", float, double)
{
  len_t n = 2;
  
  cpumat<TestType> x(n, n);
  cpumat<TestType> y(n, n);
  
  for (len_t i=0; i<n*n; i++)
    x(i) = (TestType) i+1;
  
  y.fill_linspace(1, n*n);
  REQUIRE( (x == y) );
  
  y.fill_val(1.f);
  REQUIRE( (x != y) );
}



TEMPLATE_TEST_CASE("diag", "[cpumat]", float, double)
{
  len_t m = 4;
  len_t n = 3;
  
  cpumat<TestType> x(m, n);
  x.fill_linspace(1, m*n);
  
  cpuvec<TestType> v;
  x.diag(v);
  REQUIRE( fltcmp::eq(v(0), 1) );
  REQUIRE( fltcmp::eq(v(1), 6) );
  REQUIRE( fltcmp::eq(v(2), 11) );
  
  x.antidiag(v);
  REQUIRE( fltcmp::eq(v(0), 4) );
  REQUIRE( fltcmp::eq(v(1), 7) );
  REQUIRE( fltcmp::eq(v(2), 10) );
}
