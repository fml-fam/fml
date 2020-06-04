#include "../catch.hpp"

#include <fml/gpu/card.hh>
#include <fml/gpu/gpuhelpers.hh>
#include <fml/gpu/gpuvec.hh>

using namespace arraytools;
using namespace fml;

extern std::shared_ptr<card> c;


TEMPLATE_TEST_CASE("basics - vec", "[gpuvec]", float, double)
{
  len_t n = 2;
  
  gpuvec<TestType> x(c, n);
  REQUIRE( x.size() == n );
  
  x.fill_zero();
  REQUIRE( fltcmp::eq(x.get(0), 0) );
  x.set(0, 3.14);
  REQUIRE( fltcmp::eq(x.get(0), 3.14) );
}



TEMPLATE_TEST_CASE("inheriting memory - vec", "[gpuvec]", float, double)
{
  TestType test_val;
  len_t n = 2;
  
  TestType *data = (TestType*) c->mem_alloc(n*sizeof(*data));
  
  gpuvec<TestType> x(c, data, n);
  x.fill_val(1);
  x.~gpuvec();
  c->mem_gpu2cpu(&test_val, data+0, 1*sizeof(test_val));
  REQUIRE( fltcmp::eq(test_val, 1) );
  
  gpuvec<TestType> y(c);
  y.inherit(c, data, n);
  y.fill_zero();
  y.~gpuvec();
  c->mem_gpu2cpu(&test_val, data+0, 1*sizeof(test_val));
  REQUIRE( fltcmp::eq(test_val, 0) );
  
  c->mem_free(data);
}
 


TEMPLATE_TEST_CASE("resize - vec", "[gpuvec]", float, double)
{
  len_t n = 1;
  
  gpuvec<TestType> x(c, n);
  REQUIRE( x.size() == n );
  x.fill_val(1);
  
  n = 2;
  x.resize(n);
  REQUIRE( x.size() == n );
  x.set(1, 0);
  
  REQUIRE( fltcmp::eq(x.get(0), 1) );
  REQUIRE( fltcmp::eq(x.get(1), 0) );
}



TEMPLATE_TEST_CASE("scale - vec", "[gpuvec]", float, double)
{
  len_t n = 2;
  
  gpuvec<TestType> x(c, n);
  x.fill_val(1);
  
  x.scale((TestType) 3);
  REQUIRE( fltcmp::eq(x.get(0), 3) );
  REQUIRE( fltcmp::eq(x.get(1), 3) );
}



TEMPLATE_TEST_CASE("indexing - vec", "[gpuvec]", float, double)
{
  len_t n = 2;
  
  gpuvec<TestType> x(c, n);
  gpuvec<TestType> y(c, n);
  
  for (len_t i=0; i<n; i++)
    x.set(i, i+1);
  
  y.fill_linspace(1, n);
  REQUIRE( (x == y) );
  
  y.fill_val(1.f);
  REQUIRE( (x != y) );
}



TEMPLATE_TEST_CASE("rev - vec", "[gpuvec]", float, double)
{
  len_t n = 2;
  
  gpuvec<TestType> x(c, n);
  x.fill_linspace(1, n);
  
  x.rev();
  REQUIRE( fltcmp::eq(x.get(0), 2) );
  REQUIRE( fltcmp::eq(x.get(1), 1) );
}



TEMPLATE_TEST_CASE("sum - vec", "[gpuvec]", float, double)
{
  len_t n = 5;
  
  gpuvec<TestType> x(c, n);
  x.fill_linspace(1, n);
  
  TestType s = x.sum();
  REQUIRE( fltcmp::eq(s, (n*(n+1))/2) );
}
