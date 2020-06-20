#include "../catch.hpp"

#include <fml/_internals/arraytools/src/arraytools.hpp>
#include <fml/mpi/mpimat.hh>
#include <fml/mpi/linalg.hh>

using namespace arraytools;

extern fml::grid g;


TEMPLATE_TEST_CASE("norm", "[linalg]", float, double)
{
  TestType norm;
  fml::mpimat<TestType> x(g, 3, 2, 1, 1);
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



// Current libscalapack builds on travis incorrectly lack the pXtrcon() symbols
/*
TEMPLATE_TEST_CASE("cond", "[linalg]", float, double)
{
  TestType cond;
  fml::mpimat<TestType> x(g, 3, 2, 1, 1);
  x.fill_linspace(1, 6);
  x.set(1, 0, 0);
  x.set(2, 0, 0);
  x.set(2, 1, 0);
  
  SECTION("one norm")
  {
    cond = fml::linalg::cond_1(x);
    REQUIRE( fltcmp::eq(cond, 9) );
  }
  
  SECTION("infinity norm")
  {
    cond = fml::linalg::cond_I(x);
    REQUIRE( fltcmp::eq(cond, 9) );
  }
  
  SECTION("2 norm")
  {
    // matrix from https://en.wikipedia.org/wiki/Singular_value_decomposition#Example
    x.resize(4, 5);
    x.fill_zero();
    x.set(0, 0, 1);
    x.set(3, 1, 2);
    x.set(1, 2, 3);
    x.set(0, 4, 2);
    
    cond = fml::linalg::cond_2(x);
    REQUIRE( fltcmp::eq(cond, 1.5) );
  }
}
*/
