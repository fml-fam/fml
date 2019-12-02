#include "../catch.hpp"

#include <fltcmp.hh>
#include <gpu/gpuhelpers.hh>
#include <gpu/gpumat.hh>
#include <gpu/linalg.hh>

extern std::shared_ptr<card> c;


TEMPLATE_TEST_CASE("matrix addition", "[linalg]", float, double)
{
  len_t n = 2;
  
  gpumat<TestType> x(c, n, n);
  gpumat<TestType> y(c, n, n);
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
  
  gpumat<TestType> x(c, n, n);
  gpumat<TestType> y(c, n, n);
  x.fill_linspace(1, n*n);
  y.fill_linspace(n*n, 1);
  
  gpumat<TestType> z = linalg::matmult(false, false, (TestType)1, x, y);
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
  
  gpumat<TestType> x(c, m, n);
  x.fill_linspace(1, m*n);
  
  // regular api
  gpumat<TestType> x_cp = linalg::crossprod((TestType)1, x);
  REQUIRE( x_cp.nrows() == x.ncols() );
  REQUIRE( x_cp.ncols() == x.ncols() );
  
  REQUIRE( fltcmp::eq(x_cp(0, 0), 14) );
  REQUIRE( fltcmp::eq(x_cp(1, 0), 32) );
  REQUIRE( fltcmp::eq(x_cp(1, 1), 77) );
  
  gpumat<TestType> x_tcp = linalg::tcrossprod((TestType)1, x);
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
  
  gpumat<TestType> x(c, m, n);
  x.fill_linspace(1, m*n);
  
  gpumat<TestType> tx = linalg::xpose(x);
  REQUIRE( tx.nrows() == x.ncols() );
  REQUIRE( tx.ncols() == x.nrows() );
  
  REQUIRE( fltcmp::eq(x(0, 0), tx(0, 0)) );
  REQUIRE( fltcmp::eq(x(1, 0), tx(0, 1)) );
  REQUIRE( fltcmp::eq(x(2, 1), tx(1, 2)) );
}



TEMPLATE_TEST_CASE("lu", "[linalg]", float, double)
{
  len_t n = 2;
  
  gpumat<TestType> x(c, n, n);
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
  
  cpuvec<TestType> v_cpu(n);
  v_cpu(0) = (TestType) 2;
  v_cpu(1) = (TestType) 5;
  
  gpuvec<TestType> v(c);
  gpuhelpers::cpu2gpu(v_cpu, v);
  
  gpumat<TestType> x(c, n, n);
  x.fill_diag(v);
  
  gpuvec<TestType> s(c);
  linalg::svd(x, s);
  
  v.rev();
  REQUIRE( v == s );
}



TEMPLATE_TEST_CASE("eigen", "[linalg]", float, double)
{
  len_t n = 2;
  
  cpuvec<TestType> v_cpu(n);
  v_cpu(0) = (TestType) 2;
  v_cpu(1) = (TestType) 5;
  
  gpuvec<TestType> v(c);
  gpuhelpers::cpu2gpu(v_cpu, v);
  
  gpumat<TestType> x(c, n, n);
  x.fill_diag(v);
  
  gpuvec<TestType> values(c);
  linalg::eigen(true, x, values);
  
  REQUIRE( v == values );
}



TEMPLATE_TEST_CASE("invert", "[linalg]", float, double)
{
  len_t n = 2;
  
  gpumat<TestType> x(c, n, n);
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
  
  cpuvec<TestType> y_cpu(n);
  y_cpu(0) = (TestType) 1;
  y_cpu(1) = (TestType) 1;
  
  cpumat<TestType> x_cpu(n, n);
  x_cpu.fill_zero();
  x_cpu(0, 0) = (TestType) 2;
  x_cpu(1, 1) = (TestType) 3;
  
  gpumat<TestType> x(c);
  gpuvec<TestType> y(c);
  gpuhelpers::cpu2gpu(x_cpu, x);
  gpuhelpers::cpu2gpu(y_cpu, y);
  linalg::solve(x, y);
  
  gpuhelpers::gpu2cpu(y, y_cpu);
  REQUIRE( fltcmp::eq(y_cpu(0), 0.5) );
  REQUIRE( fltcmp::eq(y_cpu(1), (TestType)1/3) );
}
