#include "../catch.hpp"

#include <fltcmp.hh>
#include <mpi/grid.hh>
#include <mpi/linalg.hh>
#include <mpi/mpimat.hh>

extern grid g;


TEMPLATE_TEST_CASE("matrix addition", "[linalg]", float, double)
{
  len_t n = 2;
  
  mpimat<TestType> x(g, n, n, 1, 1);
  mpimat<TestType> y(g, n, n, 1, 1);
  x.fill_linspace(1, n*n);
  y.fill_linspace(n*n, 1);
  
  auto z = linalg::add(false, false, (TestType)1.f, (TestType)1.f, x, y);
  
  TestType v = (TestType) n*n + 1;
  REQUIRE( fltcmp::eq(z(0, 0), v) );
  REQUIRE( fltcmp::eq(z(1, 0), v) );
  REQUIRE( fltcmp::eq(z(0, 1), v) );
  REQUIRE( fltcmp::eq(z(1, 1), v) );
  
  linalg::add(false, false, (TestType)1.f, (TestType)1.f, x, z, z);
  
  REQUIRE( fltcmp::eq(z(0, 0), v+1) );
  REQUIRE( fltcmp::eq(z(1, 0), v+2) );
  REQUIRE( fltcmp::eq(z(0, 1), v+3) );
  REQUIRE( fltcmp::eq(z(1, 1), v+4) );
  
  linalg::add(false, true, (TestType)1.f, (TestType)1.f, x, y, z);
  
  REQUIRE( fltcmp::eq(z(1, 0), v-1) );
  REQUIRE( fltcmp::eq(z(0, 1), v+1) );
  
  linalg::add(true, false, (TestType)1.f, (TestType)1.f, x, y, z);
  
  REQUIRE( fltcmp::eq(z(1, 0), v+1) );
  REQUIRE( fltcmp::eq(z(0, 1), v-1) );
}



TEMPLATE_TEST_CASE("matrix multiplication", "[linalg]", float, double)
{
  len_t n = 2;
  
  mpimat<TestType> x(g, n, n, 1, 1);
  mpimat<TestType> y(g, n, n, 1, 1);
  x.fill_linspace(1, n*n);
  y.fill_linspace(n*n, 1);
  
  mpimat<TestType> z = linalg::matmult(false, false, (TestType)1, x, y);
  REQUIRE( z.nrows() == n );
  REQUIRE( z.ncols() == n );
  
  REQUIRE( fltcmp::eq(z(0), 13) );
  REQUIRE( fltcmp::eq(z(1), 20) );
  REQUIRE( fltcmp::eq(z(2), 5) );
  REQUIRE( fltcmp::eq(z(3), 8) );
  
  linalg::matmult(true, false, (TestType)1, x, y, z);
  REQUIRE( fltcmp::eq(z(0), 10) );
  REQUIRE( fltcmp::eq(z(1), 24) );
  REQUIRE( fltcmp::eq(z(2), 4) );
  REQUIRE( fltcmp::eq(z(3), 10) );
  
  linalg::matmult(false, true, (TestType)1, x, y, z);
  REQUIRE( fltcmp::eq(z(0), 10) );
  REQUIRE( fltcmp::eq(z(1), 16) );
  REQUIRE( fltcmp::eq(z(2), 6) );
  REQUIRE( fltcmp::eq(z(3), 10) );
  
  linalg::matmult(true, true, (TestType)1, x, y, z);
  REQUIRE( fltcmp::eq(z(0), 8) );
  REQUIRE( fltcmp::eq(z(1), 20) );
  REQUIRE( fltcmp::eq(z(2), 5) );
  REQUIRE( fltcmp::eq(z(3), 13) );
}



TEMPLATE_TEST_CASE("crossprod and tcrossprod", "[linalg]", float, double)
{
  len_t m = 3;
  len_t n = 2;
  
  mpimat<TestType> x(g, m, n, 1, 1);
  x.fill_linspace(1, m*n);
  
  // regular api
  mpimat<TestType> x_cp = linalg::crossprod((TestType)1, x);
  REQUIRE( x_cp.nrows() == x.ncols() );
  REQUIRE( x_cp.ncols() == x.ncols() );
  
  REQUIRE( fltcmp::eq(x_cp(0, 0), 14) );
  REQUIRE( fltcmp::eq(x_cp(1, 0), 32) );
  REQUIRE( fltcmp::eq(x_cp(1, 1), 77) );
  
  mpimat<TestType> x_tcp = linalg::tcrossprod((TestType)1, x);
  REQUIRE( x_tcp.nrows() == x.nrows() );
  REQUIRE( x_tcp.ncols() == x.nrows() );
  
  REQUIRE( fltcmp::eq(x_tcp(0, 0), 17) );
  REQUIRE( fltcmp::eq(x_tcp(1, 0), 22) );
  REQUIRE( fltcmp::eq(x_tcp(1, 1), 29) );
  REQUIRE( fltcmp::eq(x_tcp(2, 0), 27) );
  REQUIRE( fltcmp::eq(x_tcp(2, 1), 36) );
  REQUIRE( fltcmp::eq(x_tcp(2, 2), 45) );
  
  // noalloc api
  x.fill_linspace(m*n, 1);
  
  linalg::crossprod((TestType)1, x, x_cp);
  REQUIRE( fltcmp::eq(x_cp(0, 0), 77) );
  REQUIRE( fltcmp::eq(x_cp(1, 0), 32) );
  REQUIRE( fltcmp::eq(x_cp(1, 1), 14) );
  
  linalg::tcrossprod((TestType)1, x, x_tcp);
  REQUIRE( fltcmp::eq(x_tcp(0, 0), 45) );
  REQUIRE( fltcmp::eq(x_tcp(1, 0), 36) );
  REQUIRE( fltcmp::eq(x_tcp(1, 1), 29) );
  REQUIRE( fltcmp::eq(x_tcp(2, 0), 27) );
  REQUIRE( fltcmp::eq(x_tcp(2, 1), 22) );
  REQUIRE( fltcmp::eq(x_tcp(2, 2), 17) );
}



TEMPLATE_TEST_CASE("xpose", "[linalg]", float, double)
{
  len_t m = 3;
  len_t n = 2;
  
  mpimat<TestType> x(g, m, n, 1, 1);
  x.fill_linspace(1, m*n);
  
  mpimat<TestType> tx = linalg::xpose(x);
  REQUIRE( tx.nrows() == x.ncols() );
  REQUIRE( tx.ncols() == x.nrows() );
  
  REQUIRE( fltcmp::eq(x(0, 0), tx(0, 0)) );
  REQUIRE( fltcmp::eq(x(1, 0), tx(0, 1)) );
  REQUIRE( fltcmp::eq(x(2, 1), tx(1, 2)) );
}



TEMPLATE_TEST_CASE("lu", "[linalg]", float, double)
{
  len_t n = 2;
  
  mpimat<TestType> x(g, n, n, 1, 1);
  x.fill_linspace(1, n*n);
  
  linalg::lu(x);
  
  REQUIRE( fltcmp::eq(x(0, 0), 2) );
  REQUIRE( fltcmp::eq(x(0, 1), 4) );
  REQUIRE( fltcmp::eq(x(1, 0), 0.5) );
  REQUIRE( fltcmp::eq(x(1, 1), 1) );
}



TEMPLATE_TEST_CASE("svd", "[linalg]", float, double)
{
  len_t n = 2;
  
  cpuvec<TestType> v(n);
  v(0) = (TestType) 2;
  v(1) = (TestType) 5;
  
  mpimat<TestType> x(g, n, n, 1, 1);
  x.fill_diag(v);
  
  cpuvec<TestType> s;
  linalg::svd(x, s);
  
  v.rev();
  REQUIRE( v == s );
}



TEMPLATE_TEST_CASE("eigen", "[linalg]", float, double)
{
  len_t n = 2;
  
  cpuvec<TestType> v(n);
  v(0) = (TestType) 2;
  v(1) = (TestType) 5;
  
  mpimat<TestType> x(g, n, n, 1, 1);
  x.fill_diag(v);
  
  cpuvec<TestType> values;
  linalg::eigen(true, x, values);
  
  v.rev();
  REQUIRE( v == values );
}



TEMPLATE_TEST_CASE("invert", "[linalg]", float, double)
{
  len_t n = 2;
  
  mpimat<TestType> x(g, n, n, 1, 1);
  x.fill_linspace(1, n*n);
  
  linalg::invert(x);
  
  REQUIRE( fltcmp::eq(x(0, 0), -2) );
  REQUIRE( fltcmp::eq(x(1, 0), 1) );
  REQUIRE( fltcmp::eq(x(0, 1), 1.5) );
  REQUIRE( fltcmp::eq(x(1, 1), -0.5) );
}



TEMPLATE_TEST_CASE("solve", "[linalg]", float, double)
{
  len_t n = 2;
  
  mpimat<TestType> y(g, n, n, 1, 1);
  y(0) = (TestType) 1;
  y(1) = (TestType) 1;
  
  mpimat<TestType> x(g, n, n, 1, 1);
  x.fill_zero();
  x(0, 0) = (TestType) 2;
  x(1, 1) = (TestType) 3;
  
  linalg::solve(x, y);
  
  REQUIRE( fltcmp::eq(y(0, 0), 0.5) );
  REQUIRE( fltcmp::eq(y(1, 0), (TestType)1/3) );
}
