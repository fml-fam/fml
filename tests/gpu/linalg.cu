#include "../catch.hpp"

#include <_internals/arraytools/src/arraytools.hpp>
#include <gpu/gpumat.hh>
#include <gpu/linalg.hh>

using namespace arraytools;

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
  REQUIRE( fltcmp::eq(z.get(0, 0), v) );
  REQUIRE( fltcmp::eq(z.get(1, 0), v) );
  REQUIRE( fltcmp::eq(z.get(0, 1), v) );
  REQUIRE( fltcmp::eq(z.get(1, 1), v) );
  
  linalg::add(false, false, (TestType)1.f, (TestType)1.f, x, z, z);
  
  REQUIRE( fltcmp::eq(z.get(0, 0), v+1) );
  REQUIRE( fltcmp::eq(z.get(1, 0), v+2) );
  REQUIRE( fltcmp::eq(z.get(0, 1), v+3) );
  REQUIRE( fltcmp::eq(z.get(1, 1), v+4) );
  
  linalg::add(false, true, (TestType)1.f, (TestType)1.f, x, y, z);
  
  REQUIRE( fltcmp::eq(z.get(1, 0), v-1) );
  REQUIRE( fltcmp::eq(z.get(0, 1), v+1) );
  
  linalg::add(true, false, (TestType)1.f, (TestType)1.f, x, y, z);
  
  REQUIRE( fltcmp::eq(z.get(1, 0), v+1) );
  REQUIRE( fltcmp::eq(z.get(0, 1), v-1) );
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
  
  REQUIRE( fltcmp::eq(z.get(0), 13) );
  REQUIRE( fltcmp::eq(z.get(1), 20) );
  REQUIRE( fltcmp::eq(z.get(2), 5) );
  REQUIRE( fltcmp::eq(z.get(3), 8) );
  
  linalg::matmult(true, false, (TestType)1, x, y, z);
  REQUIRE( fltcmp::eq(z.get(0), 10) );
  REQUIRE( fltcmp::eq(z.get(1), 24) );
  REQUIRE( fltcmp::eq(z.get(2), 4) );
  REQUIRE( fltcmp::eq(z.get(3), 10) );
  
  linalg::matmult(false, true, (TestType)1, x, y, z);
  REQUIRE( fltcmp::eq(z.get(0), 10) );
  REQUIRE( fltcmp::eq(z.get(1), 16) );
  REQUIRE( fltcmp::eq(z.get(2), 6) );
  REQUIRE( fltcmp::eq(z.get(3), 10) );
  
  linalg::matmult(true, true, (TestType)1, x, y, z);
  REQUIRE( fltcmp::eq(z.get(0), 8) );
  REQUIRE( fltcmp::eq(z.get(1), 20) );
  REQUIRE( fltcmp::eq(z.get(2), 5) );
  REQUIRE( fltcmp::eq(z.get(3), 13) );
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
  
  REQUIRE( fltcmp::eq(x_cp.get(0, 0), 14) );
  REQUIRE( fltcmp::eq(x_cp.get(1, 0), 32) );
  REQUIRE( fltcmp::eq(x_cp.get(1, 1), 77) );
  
  gpumat<TestType> x_tcp = linalg::tcrossprod((TestType)1, x);
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
  
  linalg::crossprod((TestType)1, x, x_cp);
  REQUIRE( fltcmp::eq(x_cp.get(0, 0), 77) );
  REQUIRE( fltcmp::eq(x_cp.get(1, 0), 32) );
  REQUIRE( fltcmp::eq(x_cp.get(1, 1), 14) );
  
  linalg::tcrossprod((TestType)1, x, x_tcp);
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
  
  gpumat<TestType> x(c, m, n);
  x.fill_linspace(1, m*n);
  
  gpumat<TestType> tx = linalg::xpose(x);
  REQUIRE( tx.nrows() == x.ncols() );
  REQUIRE( tx.ncols() == x.nrows() );
  
  REQUIRE( fltcmp::eq(x.get(0, 0), tx.get(0, 0)) );
  REQUIRE( fltcmp::eq(x.get(1, 0), tx.get(0, 1)) );
  REQUIRE( fltcmp::eq(x.get(2, 1), tx.get(1, 2)) );
}



TEMPLATE_TEST_CASE("lu", "[linalg]", float, double)
{
  len_t n = 2;
  
  gpumat<TestType> x(c, n, n);
  x.fill_linspace(1, n*n);
  
  linalg::lu(x);
  
  REQUIRE( fltcmp::eq(x.get(0, 0), 2) );
  REQUIRE( fltcmp::eq(x.get(0, 1), 4) );
  REQUIRE( fltcmp::eq(x.get(1, 0), 0.5) );
  REQUIRE( fltcmp::eq(x.get(1, 1), 1) );
}



TEMPLATE_TEST_CASE("trace", "[linalg]", float, double)
{
  len_t n = 2;
  
  gpumat<TestType> x(c, n, n);
  x.fill_linspace(1, n*n);
  
  TestType tr = linalg::trace(x);
  
  REQUIRE( fltcmp::eq(tr, 5) );
}



TEMPLATE_TEST_CASE("det", "[linalg]", float, double)
{
  len_t n = 2;
  
  gpumat<TestType> x(c, n, n);
  x.fill_linspace(1, n*n);
  
  int sign;
  TestType modulus;
  
  linalg::det(x, sign, modulus);
  REQUIRE( fltcmp::eq(sign, -1) );
  REQUIRE( fltcmp::eq(modulus, log(2.0)) );
  
  n = 4;
  x.resize(n, n);
  x.fill_linspace(1, n*n);
  
  linalg::det(x, sign, modulus);
  REQUIRE( fltcmp::eq(sign, 1) );
  REQUIRE( fltcmp::eq(sign*exp(modulus), 0) );
}



TEMPLATE_TEST_CASE("svd", "[linalg]", float, double)
{
  len_t m = 3;
  len_t n = 2;
  
  gpuvec<TestType> v(c, n);
  v.set(0, 2);
  v.set(1, 5);
  
  gpuvec<TestType> s1(c), s2(c), s3(c);
  
  gpumat<TestType> x(c, m, n);
  
  x.fill_diag(v);
  linalg::svd(x, s1);
  
  x.fill_diag(v);
  linalg::tssvd(x, s2);
  
  x.fill_diag(v);
  linalg::cpsvd(x, s3);
  
  v.rev();
  REQUIRE( v == s1 );
  REQUIRE( v == s2 );
  REQUIRE( v == s3 );
}



TEMPLATE_TEST_CASE("eigen", "[linalg]", float, double)
{
  len_t n = 2;
  
  gpuvec<TestType> v(c, n);
  v.set(0, 2);
  v.set(1, 5);
  
  gpumat<TestType> x(c, n, n);
  x.fill_diag(v);
  
  gpuvec<TestType> values(c);
  linalg::eigen_sym(x, values);
  
  REQUIRE( v == values );
}



TEMPLATE_TEST_CASE("invert", "[linalg]", float, double)
{
  len_t n = 2;
  
  gpumat<TestType> x(c, n, n);
  x.fill_linspace(1, n*n);
  
  linalg::invert(x);
  
  REQUIRE( fltcmp::eq(x.get(0, 0), -2) );
  REQUIRE( fltcmp::eq(x.get(1, 0), 1) );
  REQUIRE( fltcmp::eq(x.get(0, 1), 1.5) );
  REQUIRE( fltcmp::eq(x.get(1, 1), -0.5) );
}



TEMPLATE_TEST_CASE("solve", "[linalg]", float, double)
{
  len_t n = 2;
  
  gpuvec<TestType> y(c, n);
  y.set(0, 1);
  y.set(1, 1);
  
  gpumat<TestType> x(c, n, n);
  x.fill_zero();
  x.set(0, 0, 2);
  x.set(1, 1, 3);
  
  linalg::solve(x, y);
  
  REQUIRE( fltcmp::eq(y.get(0), 0.5) );
  REQUIRE( fltcmp::eq(y.get(1), (TestType)1/3) );
}



TEMPLATE_TEST_CASE("QR", "[linalg]", float, double)
{
  // test matrix from here https://en.wikipedia.org/wiki/QR_decomposition#Example_2
  gpumat<TestType> x(c, 3, 3);
  x.set(0, 0, 12);
  x.set(1, 0, 6);
  x.set(2, 0, -4);
  x.set(0, 1, -51);
  x.set(1, 1, 167);
  x.set(2, 1, 24);
  x.set(0, 2, 4);
  x.set(1, 2, -68);
  x.set(2, 2, -41);
  
  auto orig = x.dupe();
  
  
  
  // QR
  gpuvec<TestType> aux(c);
  linalg::qr(false, x, aux);
  
  gpumat<TestType> Q(c);
  gpuvec<TestType> work(c);
  linalg::qr_Q(x, aux, Q, work);
  
  REQUIRE( fltcmp::eq(fabs(Q.get(0, 0)), (TestType)6/7) );
  REQUIRE( fltcmp::eq(fabs(Q.get(1, 0)), (TestType)3/7) );
  REQUIRE( fltcmp::eq(fabs(Q.get(1, 1)), (TestType)158/175) );
  REQUIRE( fltcmp::eq(fabs(Q.get(1, 2)), (TestType)6/175) );
  
  gpumat<TestType> R(c);
  linalg::qr_R(x, R);
  
  REQUIRE( fltcmp::eq(fabs(R.get(0, 0)), (TestType)14) );
  REQUIRE( fltcmp::eq(fabs(R.get(1, 0)), 0) );
  REQUIRE( fltcmp::eq(fabs(R.get(0, 1)), (TestType)21) );
  REQUIRE( fltcmp::eq(fabs(R.get(1, 2)), (TestType)70) );
  
  
  
  // LQ
  // TODO: Not available in cusolver
}



TEMPLATE_TEST_CASE("chol", "[linalg]", float, double)
{
  // test matrix from here https://en.wikipedia.org/wiki/Cholesky_decomposition#Example
  gpumat<TestType> x(c, 3, 3);
  x.set(0, 0, 4);
  x.set(1, 0, 12);
  x.set(2, 0, -16);
  x.set(0, 1, 12);
  x.set(1, 1, 37);
  x.set(2, 1, -43);
  x.set(0, 2, -16);
  x.set(1, 2, -43);
  x.set(2, 2, 98);
  
  linalg::chol(x);
  
  REQUIRE( fltcmp::eq(fabs(x.get(0, 0)), (TestType)2) );
  REQUIRE( fltcmp::eq(fabs(x.get(0, 1)), (TestType)0) );
  REQUIRE( fltcmp::eq(fabs(x.get(1, 1)), (TestType)1) );
  REQUIRE( fltcmp::eq(fabs(x.get(2, 1)), (TestType)5) );
}
