#include "../catch.hpp"

#include <_internals/arraytools/src/arraytools.hpp>
#include <cpu/cpuvec.hh>

using namespace arraytools;



TEMPLATE_TEST_CASE("basics - vec", "[cpuvec]", float, double)
{
  len_t n = 2;
  
  cpuvec<TestType> x(n);
  REQUIRE( x.size() == n );
  
  x.fill_zero();
  REQUIRE( fltcmp::eq(x.get(0), 0) );
  x.set(0, 3.14);
  REQUIRE( fltcmp::eq(x.get(0), 3.14) );
}



TEMPLATE_TEST_CASE("inheriting memory - vec", "[cpuvec]", float, double)
{
  len_t n = 3;
  
  TestType *data = (TestType*) malloc(n*sizeof(*data));
  
  cpuvec<TestType> x(data, n);
  x.fill_val(1);
  x.~cpuvec();
  REQUIRE( fltcmp::eq(data[0], 1) );
  
  cpuvec<TestType> y;
  y.inherit(data, n);
  y.fill_zero();
  y.~cpuvec();
  REQUIRE( fltcmp::eq(data[0], 0) );
  
  free(data);
}
 


TEMPLATE_TEST_CASE("resize - vec", "[cpuvec]", float, double)
{
  len_t n = 1;
  
  cpuvec<TestType> x(n);
  REQUIRE( x.size() == n );
  x.fill_val(1);
  
  n = 2;
  x.resize(n);
  REQUIRE( x.size() == n );
  x.set(1, 0);
  
  REQUIRE( fltcmp::eq(x.get(0), 1) );
  REQUIRE( fltcmp::eq(x.get(1), 0) );
}



TEMPLATE_TEST_CASE("scale - vec", "[cpuvec]", float, double)
{
  len_t n = 2;
  
  cpuvec<TestType> x(n);
  x.fill_val(1);
  
  x.scale((TestType) 3);
  REQUIRE( fltcmp::eq(x.get(0), 3) );
  REQUIRE( fltcmp::eq(x.get(1), 3) );
}



TEMPLATE_TEST_CASE("indexing - vec", "[cpuvec]", float, double)
{
  len_t n = 2;
  
  cpuvec<TestType> x(n);
  cpuvec<TestType> y(n);
  
  for (len_t i=0; i<n; i++)
    x.set(i, i+1);
  
  y.fill_linspace(1, n);
  REQUIRE( (x == y) );
  
  y.fill_val(1.f);
  REQUIRE( (x != y) );
}



TEMPLATE_TEST_CASE("rev - vec", "[cpuvec]", float, double)
{
  len_t n = 2;
  
  cpuvec<TestType> x(n);
  x.fill_linspace(1, n);
  
  x.rev();
  REQUIRE( fltcmp::eq(x.get(0), 2) );
  REQUIRE( fltcmp::eq(x.get(1), 1) );
}



TEMPLATE_TEST_CASE("sum - vec", "[cpuvec]", float, double)
{
  len_t n = 5;
  
  cpuvec<TestType> x(n);
  x.fill_linspace(1, n);
  
  TestType s = x.sum();
  REQUIRE( fltcmp::eq(s, (n*(n+1))/2) );
}
