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
