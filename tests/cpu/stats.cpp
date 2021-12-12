#include "../catch.hpp"

#include <fml/_internals/arraytools/src/arraytools.hpp>

#include <fml/cpu/copy.hh>
#include <fml/cpu/stats.hh>

using namespace arraytools;


TEMPLATE_TEST_CASE("stats - pca", "[stats]", float, double)
{
  len_t m = 3;
  len_t n = 2;
  
  fml::cpumat<TestType> x(m, n);
  x.fill_linspace();
  
  fml::cpuvec<TestType> sdev;
  fml::cpumat<TestType> rot;
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



TEMPLATE_TEST_CASE("stats - cov", "[stats]", float, double)
{
  len_t m = 3;
  len_t n = 2;
  
  fml::cpumat<TestType> x(m, n);
  x.fill_linspace();
  x.set(3, 2);
  x.set(4, 0);
  x.set(5, -1);
  
  fml::cpumat<TestType> cov;
  
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
  
  fml::cpuvec<TestType> x1;
  fml::cpuvec<TestType> x2;
  x.get_col(0, x1);
  x.get_col(1, x2);
  
  TestType cv;
  cv = fml::stats::cov(x1, x2);
  REQUIRE( fltcmp::eq(cv, -1.5) );
  cv = fml::stats::cov(x2, x2);
  REQUIRE( fltcmp::eq(cv, 2+1.0/3.0) );
}



TEMPLATE_TEST_CASE("stats - cor", "[stats]", float, double)
{
  len_t m = 3;
  len_t n = 2;
  
  fml::cpumat<TestType> x(m, n);
  x.fill_linspace();
  x.set(3, 2);
  x.set(4, 0);
  x.set(5, -1);
  
  fml::cpumat<TestType> cor;
  
  // mat
  fml::stats::cor(x, cor);
  REQUIRE( fltcmp::eq(cor.get(0), 1) );
  REQUIRE( fltcmp::eq(cor.get(1), -3.0/sqrt(2.0*14.0/3.0)) );
  REQUIRE( fltcmp::eq(cor.get(3), 1) );
  
  // vec-vec
  x.fill_linspace();
  x.set(3, 2);
  x.set(4, 0);
  x.set(5, -1);
  
  fml::cpuvec<TestType> x1;
  fml::cpuvec<TestType> x2;
  x.get_col(0, x1);
  x.get_col(1, x2);
  
  TestType cr;
  cr = fml::stats::cor(x1, x2);
  REQUIRE( fltcmp::eq(cr, -3.0/sqrt(2.0*14.0/3.0)) );
  cr = fml::stats::cor(x2, x2);
  REQUIRE( fltcmp::eq(cr, 1) );
}
