#include "../catch.hpp"

#include <_internals/arraytools/src/arraytools.hpp>
#include <gpu/gpumat.hh>
#include <gpu/internals/gpu_utils.hh>

using namespace arraytools;

extern std::shared_ptr<card> c;


TEMPLATE_TEST_CASE("tri2zero - tall", "[gpu_utils]", float, double)
{
  len_t m = 5;
  len_t n = 3;
  gpumat<TestType> x(c, m, n);
  gpumat<TestType> truth(c, m, n);
  truth.fill_linspace(1, m*n);
  
  x.fill_linspace(1, m*n);
  fml::gpu_utils::tri2zero('U', false, m, n, x.data_ptr(), m);
  truth.set(0, 1, 0);
  truth.set(0, 2, 0);
  truth.set(1, 2, 0);
  REQUIRE( x == truth );
  
  x.fill_linspace(1, m*n);
  fml::gpu_utils::tri2zero('U', true, m, n, x.data_ptr(), m);
  truth.set(0, 0, 0);
  truth.set(1, 1, 0);
  truth.set(2, 2, 0);
  REQUIRE( x == truth );
  
  truth.fill_linspace(1, m*n);
  
  x.fill_linspace(1, m*n);
  fml::gpu_utils::tri2zero('L', false, n, n, x.data_ptr(), m);
  truth.set(1, 0, 0);
  truth.set(2, 0, 0);
  truth.set(2, 1, 0);
  REQUIRE( x == truth );
  
  x.fill_linspace(1, m*n);
  fml::gpu_utils::tri2zero('L', false, m, n, x.data_ptr(), m);
  truth.set(3, 0, 0);
  truth.set(4, 0, 0);
  truth.set(3, 1, 0);
  truth.set(4, 1, 0);
  truth.set(3, 2, 0);
  truth.set(4, 2, 0);
  REQUIRE( x == truth );
  
  x.fill_linspace(1, m*n);
  fml::gpu_utils::tri2zero('L', true, m, n, x.data_ptr(), m);
  truth.set(0, 0, 0);
  truth.set(1, 1, 0);
  truth.set(2, 2, 0);
  REQUIRE( x == truth );
}



TEMPLATE_TEST_CASE("tri2zero - wide", "[gpu_utils]", float, double)
{
  len_t m = 3;
  len_t n = 5;
  gpumat<TestType> x(c, m, n);
  gpumat<TestType> truth(c, m, n);
  truth.fill_linspace(1, m*n);
  
  x.fill_linspace(1, m*n);
  fml::gpu_utils::tri2zero('L', false, m, n, x.data_ptr(), m);
  truth.set(1, 0, 0);
  truth.set(2, 0, 0);
  truth.set(2, 1, 0);
  REQUIRE( x == truth );
  
  x.fill_linspace(1, m*n);
  fml::gpu_utils::tri2zero('L', true, m, n, x.data_ptr(), m);
  truth.set(0, 0, 0);
  truth.set(1, 1, 0);
  truth.set(2, 2, 0);
  REQUIRE( x == truth );
  
  truth.fill_linspace(1, m*n);
  
  x.fill_linspace(1, m*n);
  fml::gpu_utils::tri2zero('U', false, m, m, x.data_ptr(), m);
  truth.set(0, 1, 0);
  truth.set(0, 2, 0);
  truth.set(1, 2, 0);
  REQUIRE( x == truth );
  
  x.fill_linspace(1, m*n);
  fml::gpu_utils::tri2zero('U', false, m, n, x.data_ptr(), m);
  truth.set(0, 3, 0);
  truth.set(0, 4, 0);
  truth.set(1, 3, 0);
  truth.set(1, 4, 0);
  truth.set(2, 3, 0);
  truth.set(2, 4, 0);
  REQUIRE( x == truth );
  
  x.fill_linspace(1, m*n);
  fml::gpu_utils::tri2zero('U', true, m, n, x.data_ptr(), m);
  truth.set(0, 0, 0);
  truth.set(1, 1, 0);
  truth.set(2, 2, 0);
  REQUIRE( x == truth );
}
