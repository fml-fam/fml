#include "../catch.hpp"

#include <arraytools/src/arraytools.hpp>
#include <cpu/cpumat.hh>
#include <cpu/cpuvec.hh>

using namespace arraytools;



TEMPLATE_TEST_CASE("basics", "[cpumat]", float, double)
{
  len_t m = 3;
  len_t n = 2;
  
  cpumat<TestType> x(m, n);
  REQUIRE( x.nrows() == m );
  REQUIRE( x.ncols() == n );
  
  x.fill_zero();
  REQUIRE( fltcmp::eq(x.get(0, 0), 0) );
  x.set(0, 0, 3.14);
  REQUIRE( fltcmp::eq(x.get(0, 0), 3.14) );
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
  y.inherit(data, m, n);
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
  
  REQUIRE( fltcmp::eq(x.get(0), 1) );
  REQUIRE( fltcmp::eq(x.get(1), 0) );
}



TEMPLATE_TEST_CASE("scale", "[cpumat]", float, double)
{
  len_t m = 3;
  len_t n = 2;
  
  cpumat<TestType> x(m, n);
  x.fill_one();
  
  x.scale((TestType) 3);
  REQUIRE( fltcmp::eq(x.get(0), 3) );
  REQUIRE( fltcmp::eq(x.get(1), 3) );
}



TEMPLATE_TEST_CASE("indexing", "[cpumat]", float, double)
{
  len_t n = 2;
  
  cpumat<TestType> x(n, n);
  cpumat<TestType> y(n, n);
  
  for (len_t i=0; i<n*n; i++)
    x.set(i, i+1);
  
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
  REQUIRE( fltcmp::eq(v.get(0), 1) );
  REQUIRE( fltcmp::eq(v.get(1), 6) );
  REQUIRE( fltcmp::eq(v.get(2), 11) );
  
  x.antidiag(v);
  REQUIRE( fltcmp::eq(v.get(0), 4) );
  REQUIRE( fltcmp::eq(v.get(1), 7) );
  REQUIRE( fltcmp::eq(v.get(2), 10) );
}



TEMPLATE_TEST_CASE("rev", "[cpumat]", float, double)
{
  len_t n = 2;
  
  cpumat<TestType> x(n, n);
  x.fill_linspace(1, n*n);
  
  x.rev_cols();
  REQUIRE( fltcmp::eq(x.get(0, 0), 3) );
  REQUIRE( fltcmp::eq(x.get(1, 0), 4) );
  
  x.rev_rows();
  REQUIRE( fltcmp::eq(x.get(0, 0), 4) );
  REQUIRE( fltcmp::eq(x.get(1, 0), 3) );
}



TEMPLATE_TEST_CASE("get row/col", "[cpumat]", float, double)
{
  len_t n = 2;
  
  cpumat<TestType> x(n, n);
  x.fill_linspace(1, n*n);
  
  cpuvec<TestType> v;
  
  x.get_row(1, v);
  REQUIRE( fltcmp::eq(x.get(1, 0), v.get(0)) );
  REQUIRE( fltcmp::eq(x.get(1, 1), v.get(1)) );
  
  x.get_col(0, v);
  REQUIRE( fltcmp::eq(x.get(0, 0), v.get(0)) );
  REQUIRE( fltcmp::eq(x.get(1, 0), v.get(1)) );
}
