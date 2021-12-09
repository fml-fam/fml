#include "../catch.hpp"

#include <fml/_internals/arraytools/src/arraytools.hpp>
#include <fml/mpi/grid.hh>
#include <fml/mpi/linalg.hh>
#include <fml/mpi/mpimat.hh>

using namespace arraytools;

extern fml::grid g;


TEMPLATE_TEST_CASE("trace", "[linalg]", float, double)
{
  len_t n = 2;
  
  fml::mpimat<TestType> x(g, n, n, 1, 1);
  x.fill_linspace();
  
  TestType tr = fml::linalg::trace(x);
  
  REQUIRE( fltcmp::eq(tr, 5) );
}



TEMPLATE_TEST_CASE("det", "[linalg]", float, double)
{
  len_t n = 2;
  
  fml::mpimat<TestType> x(g, n, n, 1, 1);
  x.fill_linspace();
  
  int sign;
  TestType modulus;
  
  fml::linalg::det(x, sign, modulus);
  REQUIRE( fltcmp::eq(sign, -1) );
  REQUIRE( fltcmp::eq(modulus, log(2.0)) );
  
  n = 4;
  x.resize(n, n);
  x.fill_linspace();
  
  fml::linalg::det(x, sign, modulus);
  REQUIRE( fltcmp::eq(sign, 1) );
  REQUIRE( fltcmp::eq(sign*exp(modulus), 0) );
}
