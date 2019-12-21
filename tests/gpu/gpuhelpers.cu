#include "../catch.hpp"

#include <arraytools/src/arraytools.hpp>
#include <cpu/cpumat.hh>
#include <gpu/card.hh>
#include <gpu/gpuhelpers.hh>
#include <gpu/gpumat.hh>

using namespace arraytools;

extern std::shared_ptr<card> c;


TEMPLATE_TEST_CASE("gpu2cpu", "[gpumat]", float, double)
{
  len_t m = 3;
  len_t n = 2;
  
  gpumat<TestType> x(c, m, n);
  cpumat<TestType> x_true(m, n);
  
  x.fill_linspace(1, m*n);
  x_true.fill_linspace(1, m*n);
  cpumat<TestType> x_test = gpuhelpers::gpu2cpu(x);
  REQUIRE( x_test == x_true );
  
  x.fill_eye();
  gpuhelpers::gpu2cpu(x, x_test);
  x_true.fill_eye();
  REQUIRE( x_test == x_true );
}



TEMPLATE_TEST_CASE("cpu2gpu", "[gpumat]", float, double)
{
  len_t m = 3;
  len_t n = 2;
  
  gpumat<TestType> x_true(c, m, n);
  x_true.fill_linspace(1, m*n);
  
  cpumat<TestType> x(m, n);
  x.fill_linspace(1, m*n);
  
  gpumat<TestType> x_test(c);
  gpuhelpers::cpu2gpu(x, x_test);
  
  REQUIRE( x_test == x_true );
}



TEMPLATE_TEST_CASE("gpu2gpu", "[gpumat]", float, double)
{
  len_t m = 3;
  len_t n = 2;
  
  gpumat<TestType> x_test(c);
  gpumat<TestType> x_true(c, m, n);
  x_true.fill_linspace(1, m*n);
  
  gpuhelpers::gpu2gpu(x_true, x_test);
  
  REQUIRE( x_test == x_true );
}