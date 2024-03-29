#include "../catch.hpp"

#include <fml/_internals/arraytools/src/arraytools.hpp>
#include <fml/mpi/internals/bcutils.hh>
#include <fml/mpi/copy.hh>
#include <fml/mpi/grid.hh>
#include <fml/mpi/mpimat.hh>

using namespace arraytools;

extern fml::grid g;


TEMPLATE_TEST_CASE("basics", "[mpimat]", float, double)
{
  len_t m = 3;
  len_t n = 2;
  
  fml::mpimat<TestType> x(g, m, n, 1, 1);
  REQUIRE( x.nrows() == m );
  REQUIRE( x.ncols() == n );
  
  x.fill_zero();
  REQUIRE( fltcmp::eq(x.get(0, 0), 0) );
  x.set(0, 0, 3.14);
  REQUIRE( fltcmp::eq(x.get(0, 0), 3.14) );
}



TEMPLATE_TEST_CASE("inheriting memory", "[mpimat]", float, double)
{
  TestType testval;
  
  len_t m = 6;
  len_t n = 5;
  int mb = 2;
  int nb = 2;
  
  len_local_t m_local = fml::bcutils::numroc(m, mb, g.myrow(), 0, g.nprow());
  len_local_t n_local = fml::bcutils::numroc(n, nb, g.mycol(), 0, g.npcol());
  
  TestType *data = (TestType*) malloc(m_local*n_local*sizeof(*data));
  
  fml::mpimat<TestType> x(g, data, m, n, mb, nb);
  x.fill_eye();
  x.~mpimat();
  
  if (g.rank0())
    testval = data[0];
  
  g.bcast(1, 1, &testval, 'A', 0, 0);
  REQUIRE( fltcmp::eq(testval, 1) );
  
  fml::mpimat<TestType> y(g);
  y.inherit(g, data, m, n, mb, nb);
  y.fill_zero();
  y.~mpimat();
  
  if (g.rank0())
    testval = data[0];
  
  g.bcast(1, 1, &testval, 'A', 0, 0);
  REQUIRE( fltcmp::eq(testval, 0) );
  
  free(data);
}



TEMPLATE_TEST_CASE("resize", "[mpimat]", float, double)
{
  len_t m = 6;
  len_t n = 5;
  int mb = 2;
  int nb = 2;
  
  fml::mpimat<TestType> x(g);
  x.resize(m, n, mb, nb);
  x.fill_eye();
  
  REQUIRE( x.nrows() == m );
  REQUIRE( x.ncols() == n );
  
  REQUIRE( fltcmp::eq(x.get(0), 1) );
  REQUIRE( fltcmp::eq(x.get(1), 0) );
}



TEMPLATE_TEST_CASE("scale", "[mpimat]", float, double)
{
  len_t m = 6;
  len_t n = 5;
  int mb = 2;
  int nb = 2;

  fml::mpimat<float> x(g, m, n, mb, nb);
  x.fill_val(1);

  x.scale(3.0f);
  REQUIRE( fltcmp::eq(x.get(0), 3) );
  REQUIRE( fltcmp::eq(x.get(1), 3) );
}



TEMPLATE_TEST_CASE("indexing", "[mpimat]", float, double)
{
  len_t n = 2;
  
  fml::cpumat<TestType> x_cpu = fml::cpumat<TestType>(n, n);
  
  TestType *x_d = x_cpu.data_ptr();
  
  for (len_t i=0; i<n*n; i++)
    x_d[i] = (TestType) i+1;
  
  fml::mpimat<TestType> x(g, 1, 1);
  fml::copy::cpu2mpi(x_cpu, x);
  fml::mpimat<TestType> y(g, n, n, 1, 1);
  
  y.fill_linspace();
  REQUIRE( (x == y) );
  
  y.fill_val(1.f);
  REQUIRE( (x != y) );
}



TEMPLATE_TEST_CASE("diag", "[cpumat]", float, double)
{
  len_t m = 4;
  len_t n = 3;
  
  fml::mpimat<TestType> x(g, m, n, 1, 1);
  x.fill_linspace();
  
  fml::cpuvec<TestType> v;
  x.diag(v);
  REQUIRE( fltcmp::eq(v.get(0), 1) );
  REQUIRE( fltcmp::eq(v.get(1), 6) );
  REQUIRE( fltcmp::eq(v.get(2), 11) );
  
  x.antidiag(v);
  REQUIRE( fltcmp::eq(v.get(0), 4) );
  REQUIRE( fltcmp::eq(v.get(1), 7) );
  REQUIRE( fltcmp::eq(v.get(2), 10) );
}



TEMPLATE_TEST_CASE("rev", "[cpumat]", float, double)
{
  len_t n = 2;
  
  fml::mpimat<TestType> x(g, n, n, 1, 1);
  x.fill_linspace();
  
  x.rev_cols();
  REQUIRE( fltcmp::eq(x.get(0, 0), 3) );
  REQUIRE( fltcmp::eq(x.get(1, 0), 4) );
  
  // 4, 3 not 2, 1 because cols are still reversed from above
  x.rev_rows();
  REQUIRE( fltcmp::eq(x.get(0, 0), 4) );
  REQUIRE( fltcmp::eq(x.get(1, 0), 3) );
}



TEMPLATE_TEST_CASE("get row/col", "[cpumat]", float, double)
{
  len_t n = 2;
  
  fml::mpimat<TestType> x(g, n, n, 1, 1);
  x.fill_linspace();
  
  fml::cpuvec<TestType> v;
  
  x.get_row(1, v);
  REQUIRE( fltcmp::eq(x.get(1, 0), v.get(0)) );
  REQUIRE( fltcmp::eq(x.get(1, 1), v.get(1)) );
  
  x.get_col(0, v);
  REQUIRE( fltcmp::eq(x.get(0, 0), v.get(0)) );
  REQUIRE( fltcmp::eq(x.get(1, 0), v.get(1)) );
}



TEMPLATE_TEST_CASE("set row/col", "[cpumat]", float, double)
{
  len_t n = 2;
  
  fml::mpimat<TestType> x(g, n, n, 1, 1);
  x.fill_linspace();
  
  fml::cpuvec<TestType> v(n);
  v.fill_linspace(9, 8);
  
  x.set_row(1, v);
  REQUIRE( fltcmp::eq(x.get(1, 0), v.get(0)) );
  REQUIRE( fltcmp::eq(x.get(1, 1), v.get(1)) );
  
  x.set_col(0, v);
  REQUIRE( fltcmp::eq(x.get(0, 0), v.get(0)) );
  REQUIRE( fltcmp::eq(x.get(1, 0), v.get(1)) );
}
