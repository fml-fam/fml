#include "../catch.hpp"

#include <arraytools/src/arraytools.hpp>
#include <cpu/cpumat.hh>
#include <gpu/card.hh>
#include <gpu/gpuhelpers.hh>
#include <gpu/gpumat.hh>

using namespace arraytools;

extern std::shared_ptr<card> c;


TEMPLATE_TEST_CASE("basics", "[gpumat]", float, double)
{
  len_t m = 3;
  len_t n = 2;
  gpumat<TestType> x(c, m, n);
  
  REQUIRE( x.nrows() == m );
  REQUIRE( x.ncols() == n );
  
  x.fill_zero();
  REQUIRE( fltcmp::eq(x.get(0, 0), 0) );
  x.set(0, 0, 3.14);
  REQUIRE( fltcmp::eq(x.get(0, 0), 3.14) );
}



TEMPLATE_TEST_CASE("inheriting memory", "[gpumat]", float, double)
{
  TestType test_val;
  len_t m = 2;
  len_t n = 3;
  
  TestType *data = (TestType*) c->mem_alloc(m*n*sizeof(*data));
  
  gpumat<TestType> x(c, data, m, n);
  x.fill_eye();
  x.~gpumat();
  c->mem_gpu2cpu(&test_val, data+0, 1*sizeof(test_val));
  REQUIRE( fltcmp::eq(test_val, 1) );
  
  gpumat<TestType> y(c);
  y.inherit(c, data, m, n);
  y.fill_zero();
  y.~gpumat();
  c->mem_gpu2cpu(&test_val, data+0, 1*sizeof(test_val));
  REQUIRE( fltcmp::eq(test_val, 0) );
  
  c->mem_free(data);
}



TEMPLATE_TEST_CASE("resize", "[gpumat]", float, double)
{
  len_t m = 3;
  len_t n = 2;
  
  gpumat<TestType> x(c);
  x.resize(n, m);
  x.fill_eye();
  
  REQUIRE( x.nrows() == n );
  REQUIRE( x.ncols() == m );
  
  REQUIRE( fltcmp::eq(x.get(0), 1) );
  REQUIRE( fltcmp::eq(x.get(1), 0) );
}



TEMPLATE_TEST_CASE("scale", "[gpumat]", float, double)
{
  len_t m = 3;
  len_t n = 2;
  
  gpumat<TestType> x(c, m, n);
  x.fill_val(1);
  
  x.scale((TestType) 3);
  REQUIRE( fltcmp::eq(x.get(0), 3) );
  REQUIRE( fltcmp::eq(x.get(1), 3) );
}



TEMPLATE_TEST_CASE("indexing", "[gpumat]", float, double)
{
  len_t n = 2;

  gpumat<TestType> x(c, n, n);
  gpumat<TestType> y(c, n, n);

  for (len_t i=0; i<n*n; i++)
    x.set(i, (TestType) i+1);

  y.fill_linspace(1.f, (TestType) n*n);
  REQUIRE( (x == y) );

  y.fill_val(1.f);
  REQUIRE( (x != y) );
}



TEMPLATE_TEST_CASE("diag", "[gpumat]", float, double)
{
  len_t m = 4;
  len_t n = 3;
  
  gpumat<TestType> x(c, m, n);
  x.fill_linspace(1, m*n);
  
  gpuvec<TestType> v(c);
  x.diag(v);
  REQUIRE( fltcmp::eq(v.get(0), 1) );
  REQUIRE( fltcmp::eq(v.get(1), 6) );
  REQUIRE( fltcmp::eq(v.get(2), 11) );
  
  x.antidiag(v);
  REQUIRE( fltcmp::eq(v.get(0), 4) );
  REQUIRE( fltcmp::eq(v.get(1), 7) );
  REQUIRE( fltcmp::eq(v.get(2), 10) );
}
