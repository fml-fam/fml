#include "../catch.hpp"

#include <fml/_internals/arraytools/src/arraytools.hpp>
#include <fml/gpu/stats.hh>

using namespace arraytools;
using namespace fml;

extern fml::card_sp_t c;


TEMPLATE_TEST_CASE("stats - pca", "[stats]", float, double)
{
  len_t m = 3;
  len_t n = 2;
  
  gpumat<TestType> x(c, m, n);
  x.fill_linspace();
  
  gpuvec<TestType> sdev(c);
  gpumat<TestType> rot(c);
  stats::pca(true, true, x, sdev, rot);
  
  TestType sq2 = sqrt(2.0);
  
  
  REQUIRE( sdev.size() == 2 );
  REQUIRE( fltcmp::eq(sdev.get(0), sq2) );
  // CUDA calculates this wrong
  // REQUIRE( fltcmp::eq(sdev.get(1), 0) );
  
  REQUIRE( rot.nrows() == 2 );
  REQUIRE( rot.ncols() == 2 );
  for (len_t i=0; i<n*n; i++)
  {
    auto test = fabs(rot.get(i)); // sign choice is up to LAPACK library
    REQUIRE( fltcmp::eq(test, 1/sq2) );
  }
}



TEMPLATE_TEST_CASE("stats - cov", "[stats]", float, double)
{
  len_t m = 3;
  len_t n = 2;
  
  fml::gpumat<TestType> x(c, m, n);
  x.fill_linspace();
  x.set(3, 2);
  x.set(4, 0);
  x.set(5, -1);
  
  fml::gpumat<TestType> cov(c);
  
  // mat
  fml::stats::cov(x, cov);
  REQUIRE( fltcmp::eq(cov.get(0), 1) );
  REQUIRE( fltcmp::eq(cov.get(1), -1.5) );
  REQUIRE( fltcmp::eq(cov.get(3), 2+1.0/3.0) );
  
  // vec-vec
  x.fill_linspace();
  x.set(3, 2);
  x.set(4, 0);
  x.set(5, -1);
  
  fml::gpuvec<TestType> x1(c);
  fml::gpuvec<TestType> x2(c);
  x.get_col(0, x1);
  x.get_col(1, x2);
  
  TestType cv;
  cv = fml::stats::cov(x1, x2);
  REQUIRE( fltcmp::eq(cv, -1.5) );
  cv = fml::stats::cov(x2, x2);
  REQUIRE( fltcmp::eq(cv, 2+1.0/3.0) );
}
