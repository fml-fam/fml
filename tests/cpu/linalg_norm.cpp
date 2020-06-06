#include "../catch.hpp"

#include <fml/_internals/arraytools/src/arraytools.hpp>
#include <fml/cpu/cpumat.hh>
#include <fml/cpu/cpuvec.hh>
#include <fml/cpu/linalg.hh>

using namespace arraytools;


TEMPLATE_TEST_CASE("norm", "[linalg]", float, double)
{
  TestType norm;
  fml::cpumat<TestType> x(3, 2);
  x.fill_linspace(1, 6);
  
  SECTION("one norm")
  {
    norm = fml::linalg::norm_1(x);
    REQUIRE( fltcmp::eq(norm, 15) );
  }
  
  SECTION("infinity norm")
  {
    norm = fml::linalg::norm_I(x);
    REQUIRE( fltcmp::eq(norm, 9) );
  }
  
  SECTION("frobenius norm")
  {
    norm = fml::linalg::norm_F(x);
    REQUIRE( fltcmp::eq(norm, sqrtf(91)) );
  }
  
  SECTION("max-mod norm")
  {
    norm = fml::linalg::norm_M(x);
    REQUIRE( fltcmp::eq(norm, 6) );
  }
}



TEMPLATE_TEST_CASE("norm_2", "[linalg]", float, double)
{
  // matrix from https://en.wikipedia.org/wiki/Singular_value_decomposition#Example
  TestType norm;
  fml::cpumat<TestType> x(4, 5);
  x.fill_zero();
  x.set(0, 0, 1);
  x.set(3, 1, 2);
  x.set(1, 2, 3);
  x.set(0, 4, 2);
  
  norm = fml::linalg::norm_2(x);
  REQUIRE( fltcmp::eq(norm, 3) );
}
