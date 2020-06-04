#include "../catch.hpp"

#include <fml/_internals/arraytools/src/arraytools.hpp>
#include <fml/mpi/stats.hh>

using namespace arraytools;

extern fml::grid g;


TEMPLATE_TEST_CASE("stats - pca", "[stats]", float, double)
{
  len_t m = 3;
  len_t n = 2;
  
  fml::mpimat<TestType> x(g, m, n, 1, 1);
  x.fill_linspace(1, m*n);
  
  fml::cpuvec<TestType> sdev;
  fml::mpimat<TestType> rot(g, 1, 1);
  fml::stats::pca(true, true, x, sdev, rot);
  
  TestType sq2 = sqrt(2.0);
  
  REQUIRE( sdev.size() == 2 );
  REQUIRE( fltcmp::eq(sdev.get(0), sq2) );
  REQUIRE( fltcmp::eq(sdev.get(1), 0) );
  
  REQUIRE( rot.nrows() == 2 );
  REQUIRE( rot.ncols() == 2 );
  for (len_t i=0; i<n*n; i++)
  {
    auto test = fabs(rot.get(i)); // sign choice is up to LAPACK library
    REQUIRE( fltcmp::eq(test, 1/sq2) );
  }
}
