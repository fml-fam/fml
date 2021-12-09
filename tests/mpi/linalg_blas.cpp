#include "../catch.hpp"

#include <fml/_internals/arraytools/src/arraytools.hpp>
#include <fml/mpi/grid.hh>
#include <fml/mpi/linalg.hh>
#include <fml/mpi/mpimat.hh>

using namespace arraytools;

extern fml::grid g;


TEMPLATE_TEST_CASE("matrix addition", "[linalg]", float, double)
{
  len_t n = 2;
  
  fml::mpimat<TestType> x(g, n, n, 1, 1);
  fml::mpimat<TestType> y(g, n, n, 1, 1);
  x.fill_linspace(1, n*n);
  y.fill_linspace(n*n, 1);
  
  auto z = fml::linalg::add(false, false, (TestType)1.f, (TestType)1.f, x, y);
  
  TestType v = (TestType) n*n + 1;
  REQUIRE( fltcmp::eq(z.get(0, 0), v) );
  REQUIRE( fltcmp::eq(z.get(1, 0), v) );
  REQUIRE( fltcmp::eq(z.get(0, 1), v) );
  REQUIRE( fltcmp::eq(z.get(1, 1), v) );
  
  fml::linalg::add(false, false, (TestType)1.f, (TestType)1.f, x, z, z);
  
  REQUIRE( fltcmp::eq(z.get(0, 0), v+1) );
  REQUIRE( fltcmp::eq(z.get(1, 0), v+2) );
  REQUIRE( fltcmp::eq(z.get(0, 1), v+3) );
  REQUIRE( fltcmp::eq(z.get(1, 1), v+4) );
  
  fml::linalg::add(false, true, (TestType)1.f, (TestType)1.f, x, y, z);
  
  REQUIRE( fltcmp::eq(z.get(1, 0), v-1) );
  REQUIRE( fltcmp::eq(z.get(0, 1), v+1) );
  
  fml::linalg::add(true, false, (TestType)1.f, (TestType)1.f, x, y, z);
  
  REQUIRE( fltcmp::eq(z.get(1, 0), v+1) );
  REQUIRE( fltcmp::eq(z.get(0, 1), v-1) );
}



TEMPLATE_TEST_CASE("matrix multiplication", "[linalg]", float, double)
{
  len_t n = 2;
  
  fml::mpimat<TestType> x(g, n, n, 1, 1);
  fml::mpimat<TestType> y(g, n, n, 1, 1);
  x.fill_linspace(1, n*n);
  y.fill_linspace(n*n, 1);
  
  fml::mpimat<TestType> z = fml::linalg::matmult(false, false, (TestType)1, x, y);
  REQUIRE( z.nrows() == n );
  REQUIRE( z.ncols() == n );
  
  REQUIRE( fltcmp::eq(z.get(0), 13) );
  REQUIRE( fltcmp::eq(z.get(1), 20) );
  REQUIRE( fltcmp::eq(z.get(2), 5) );
  REQUIRE( fltcmp::eq(z.get(3), 8) );
  
  fml::linalg::matmult(true, false, (TestType)1, x, y, z);
  REQUIRE( fltcmp::eq(z.get(0), 10) );
  REQUIRE( fltcmp::eq(z.get(1), 24) );
  REQUIRE( fltcmp::eq(z.get(2), 4) );
  REQUIRE( fltcmp::eq(z.get(3), 10) );
  
  fml::linalg::matmult(false, true, (TestType)1, x, y, z);
  REQUIRE( fltcmp::eq(z.get(0), 10) );
  REQUIRE( fltcmp::eq(z.get(1), 16) );
  REQUIRE( fltcmp::eq(z.get(2), 6) );
  REQUIRE( fltcmp::eq(z.get(3), 10) );
  
  fml::linalg::matmult(true, true, (TestType)1, x, y, z);
  REQUIRE( fltcmp::eq(z.get(0), 8) );
  REQUIRE( fltcmp::eq(z.get(1), 20) );
  REQUIRE( fltcmp::eq(z.get(2), 5) );
  REQUIRE( fltcmp::eq(z.get(3), 13) );
}



TEMPLATE_TEST_CASE("crossprod and tcrossprod", "[linalg]", float, double)
{
  len_t m = 3;
  len_t n = 2;
  
  fml::mpimat<TestType> x(g, m, n, 1, 1);
  x.fill_linspace();
  
  // regular api
  fml::mpimat<TestType> x_cp = fml::linalg::crossprod((TestType)1, x);
  REQUIRE( x_cp.nrows() == x.ncols() );
  REQUIRE( x_cp.ncols() == x.ncols() );
  
  REQUIRE( fltcmp::eq(x_cp.get(0, 0), 14) );
  REQUIRE( fltcmp::eq(x_cp.get(1, 0), 32) );
  REQUIRE( fltcmp::eq(x_cp.get(1, 1), 77) );
  
  fml::mpimat<TestType> x_tcp = fml::linalg::tcrossprod((TestType)1, x);
  REQUIRE( x_tcp.nrows() == x.nrows() );
  REQUIRE( x_tcp.ncols() == x.nrows() );
  
  REQUIRE( fltcmp::eq(x_tcp.get(0, 0), 17) );
  REQUIRE( fltcmp::eq(x_tcp.get(1, 0), 22) );
  REQUIRE( fltcmp::eq(x_tcp.get(1, 1), 29) );
  REQUIRE( fltcmp::eq(x_tcp.get(2, 0), 27) );
  REQUIRE( fltcmp::eq(x_tcp.get(2, 1), 36) );
  REQUIRE( fltcmp::eq(x_tcp.get(2, 2), 45) );
  
  // noalloc api
  x.fill_linspace(m*n, 1);
  
  fml::linalg::crossprod((TestType)1, x, x_cp);
  REQUIRE( fltcmp::eq(x_cp.get(0, 0), 77) );
  REQUIRE( fltcmp::eq(x_cp.get(1, 0), 32) );
  REQUIRE( fltcmp::eq(x_cp.get(1, 1), 14) );
  
  fml::linalg::tcrossprod((TestType)1, x, x_tcp);
  REQUIRE( fltcmp::eq(x_tcp.get(0, 0), 45) );
  REQUIRE( fltcmp::eq(x_tcp.get(1, 0), 36) );
  REQUIRE( fltcmp::eq(x_tcp.get(1, 1), 29) );
  REQUIRE( fltcmp::eq(x_tcp.get(2, 0), 27) );
  REQUIRE( fltcmp::eq(x_tcp.get(2, 1), 22) );
  REQUIRE( fltcmp::eq(x_tcp.get(2, 2), 17) );
}



TEMPLATE_TEST_CASE("xpose", "[linalg]", float, double)
{
  len_t m = 3;
  len_t n = 2;
  
  fml::mpimat<TestType> x(g, m, n, 1, 1);
  x.fill_linspace();
  
  fml::mpimat<TestType> tx = fml::linalg::xpose(x);
  REQUIRE( tx.nrows() == x.ncols() );
  REQUIRE( tx.ncols() == x.nrows() );
  
  REQUIRE( fltcmp::eq(x.get(0, 0), tx.get(0, 0)) );
  REQUIRE( fltcmp::eq(x.get(1, 0), tx.get(0, 1)) );
  REQUIRE( fltcmp::eq(x.get(2, 1), tx.get(1, 2)) );
}
