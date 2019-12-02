#include "../catch.hpp"

#include <fltcmp.hh>
#include <cpu/cpumat.hh>
#include <gpu/card.hh>
#include <gpu/gpuhelpers.hh>
#include <gpu/gpumat.hh>


extern std::shared_ptr<card> c;


TEMPLATE_TEST_CASE("basics", "[gpumat]", float, double)
{
  len_t m = 3;
  len_t n = 2;
  gpumat<TestType> x(c, m, n);
  
  REQUIRE( x.nrows() == m );
  REQUIRE( x.ncols() == n );
  
  x.fill_zero();
  // REQUIRE( fltcmp::eq(x(0, 0), 0) );
  // x(0, 0) = (TestType) 3.14;
  // REQUIRE( fltcmp::eq(x(0, 0), 3.14) );
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
  y.set(c, data, m, n);
  y.fill_zero();
  y.~gpumat();
  c->mem_gpu2cpu(&test_val, data+0, 1*sizeof(test_val));
  REQUIRE( fltcmp::eq(test_val, 0) );
  
  c->mem_free(data);
}
