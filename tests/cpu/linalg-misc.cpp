#include "../catch.hpp"

#include <fml/_internals/arraytools/src/arraytools.hpp>
#include <fml/cpu/cpumat.hh>
#include <fml/cpu/cpuvec.hh>
#include <fml/cpu/linalg.hh>

using namespace arraytools;


TEMPLATE_TEST_CASE("trace", "[linalg]", float, double)
{
  len_t n = 2;
  
  fml::cpumat<TestType> x(n, n);
  x.fill_linspace(1, n*n);
  
  TestType tr = fml::linalg::trace(x);
  
  REQUIRE( fltcmp::eq(tr, 5) );
}



TEMPLATE_TEST_CASE("det", "[linalg]", float, double)
{
  len_t n = 2;
  
  fml::cpumat<TestType> x(n, n);
  x.fill_linspace(1, n*n);
  
  int sign;
  TestType modulus;
  
  fml::linalg::det(x, sign, modulus);
  REQUIRE( fltcmp::eq(sign, -1) );
  REQUIRE( fltcmp::eq(modulus, log(2.0)) );
  
  n = 4;
  x.resize(n, n);
  x.fill_linspace(1, n*n);
  
  fml::linalg::det(x, sign, modulus);
  REQUIRE( fltcmp::eq(sign, 1) );
  REQUIRE( fltcmp::eq(sign*exp(modulus), 0) );
}
