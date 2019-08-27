#include "../catch.hpp"
#include "../fltcmp.hh"

#include <cpu/cpumat.hh>
#include <cpu/linalg.hh>



TEMPLATE_TEST_CASE("matrix multiplication", "[linalg]", float, double)
{
  len_t n = 2;
  
  cpumat<TestType> x(n, n);
  cpumat<TestType> y(n, n);
  
  x.fill_linspace(1.f, (float) n*n);
  y.fill_linspace((float) n*n, 1.f);
  
  cpumat<TestType> z = linalg::matmult(false, false, (TestType)1, x, y);
  REQUIRE( z.nrows() == n );
  REQUIRE( z.ncols() == n );
  
  REQUIRE( fltcmp::eq(z(0), 13) );
  REQUIRE( fltcmp::eq(z(1), 20) );
  REQUIRE( fltcmp::eq(z(2), 5) );
  REQUIRE( fltcmp::eq(z(3), 8) );
  
  linalg::matmult_noalloc(true, false, (TestType)1, x, y, z);
  REQUIRE( fltcmp::eq(z(0), 10) );
  REQUIRE( fltcmp::eq(z(1), 24) );
  REQUIRE( fltcmp::eq(z(2), 4) );
  REQUIRE( fltcmp::eq(z(3), 10) );
  
  linalg::matmult_noalloc(false, true, (TestType)1, x, y, z);
  REQUIRE( fltcmp::eq(z(0), 10) );
  REQUIRE( fltcmp::eq(z(1), 16) );
  REQUIRE( fltcmp::eq(z(2), 6) );
  REQUIRE( fltcmp::eq(z(3), 10) );
  
  linalg::matmult_noalloc(true, true, (TestType)1, x, y, z);
  REQUIRE( fltcmp::eq(z(0), 8) );
  REQUIRE( fltcmp::eq(z(1), 20) );
  REQUIRE( fltcmp::eq(z(2), 5) );
  REQUIRE( fltcmp::eq(z(3), 13) );
}



TEMPLATE_TEST_CASE("crossprod and tcrossprod", "[linalg]", float, double)
{
  len_t m = 3;
  len_t n = 2;
  
  cpumat<TestType> x(m, n);
  x.fill_linspace(1.f, (TestType) m*n);
  
  // regular api
  cpumat<TestType> x_cp = linalg::crossprod((TestType)1, x);
  REQUIRE( x_cp.nrows() == x.ncols() );
  REQUIRE( x_cp.ncols() == x.ncols() );
  
  REQUIRE( fltcmp::eq(x_cp(0, 0), 14) );
  REQUIRE( fltcmp::eq(x_cp(1, 0), 32) );
  REQUIRE( fltcmp::eq(x_cp(1, 1), 77) );
  
  cpumat<TestType> x_tcp = linalg::tcrossprod((TestType)1, x);
  REQUIRE( x_tcp.nrows() == x.nrows() );
  REQUIRE( x_tcp.ncols() == x.nrows() );
  
  REQUIRE( fltcmp::eq(x_tcp(0, 0), 17) );
  REQUIRE( fltcmp::eq(x_tcp(1, 0), 22) );
  REQUIRE( fltcmp::eq(x_tcp(1, 1), 29) );
  REQUIRE( fltcmp::eq(x_tcp(2, 0), 27) );
  REQUIRE( fltcmp::eq(x_tcp(2, 1), 36) );
  REQUIRE( fltcmp::eq(x_tcp(2, 2), 45) );
  
  // noalloc api
  x.fill_linspace((TestType) m*n, 1.f);
  
  linalg::crossprod_noalloc((TestType)1, x, x_cp);
  REQUIRE( fltcmp::eq(x_cp(0, 0), 77) );
  REQUIRE( fltcmp::eq(x_cp(1, 0), 32) );
  REQUIRE( fltcmp::eq(x_cp(1, 1), 14) );
  
  linalg::tcrossprod_noalloc((TestType)1, x, x_tcp);
  REQUIRE( fltcmp::eq(x_tcp(0, 0), 45) );
  REQUIRE( fltcmp::eq(x_tcp(1, 0), 36) );
  REQUIRE( fltcmp::eq(x_tcp(1, 1), 29) );
  REQUIRE( fltcmp::eq(x_tcp(2, 0), 27) );
  REQUIRE( fltcmp::eq(x_tcp(2, 1), 22) );
  REQUIRE( fltcmp::eq(x_tcp(2, 2), 17) );
}
