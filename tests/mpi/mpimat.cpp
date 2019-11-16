#include "../catch.hpp"

#include <fltcmp.hh>
#include <mpi/bcutils.hh>
#include <mpi/grid.hh>
#include <mpi/mpihelpers.hh>
#include <mpi/mpimat.hh>


extern grid g;



TEMPLATE_TEST_CASE("basics", "[mpimat]", float, double)
{
  len_t m = 3;
  len_t n = 2;
  
  mpimat<TestType> x(g, m, n, 1, 1);
  REQUIRE( x.nrows() == m );
  REQUIRE( x.ncols() == n );
  
  x.fill_zero();
  REQUIRE( fltcmp::eq(x(0, 0), 0) );
  // FIXME TODO can tis be fixed to match cpumat insertion?
  // x(0, 0) = (TestType) 3.14;
  if (g.rank0())
    x.data_ptr()[0] = (TestType) 3.14;
  
  REQUIRE( fltcmp::eq(x(0, 0), 3.14) );
}



TEMPLATE_TEST_CASE("inheriting memory", "[mpimat]", float, double)
{
  TestType testval;
  
  len_t m = 6;
  len_t n = 5;
  int mb = 2;
  int nb = 2;
  
  len_local_t m_local = bcutils::numroc(m, mb, g.myrow(), 0, g.nprow());
  len_local_t n_local = bcutils::numroc(n, nb, g.mycol(), 0, g.npcol());
  
  TestType *data = (TestType*) malloc(m_local*n_local*sizeof(*data));
  
  mpimat<TestType> x(g, data, m, n, mb, nb);
  x.fill_eye();
  x.~mpimat();
  
  if (g.rank0())
    testval = data[0];
  
  g.bcast(1, 1, &testval, 'A', 0, 0);
  REQUIRE( fltcmp::eq(testval, 1) );
  
  mpimat<TestType> y(g);
  y.set(g, data, m, n, mb, nb);
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
  
  mpimat<TestType> x(g);
  x.resize(m, n, mb, nb);
  x.fill_eye();
  
  REQUIRE( x.nrows() == m );
  REQUIRE( x.ncols() == n );
  
  REQUIRE( fltcmp::eq(x(0), 1) );
  REQUIRE( fltcmp::eq(x(1), 0) );
}



TEMPLATE_TEST_CASE("scale", "[mpimat]", float, double)
{
  len_t m = 6;
  len_t n = 5;
  int mb = 2;
  int nb = 2;

  mpimat<float> x(g, m, n, mb, nb);
  x.fill_one();

  x.scale(3.0f);
  REQUIRE( fltcmp::eq(x(0), 3) );
  REQUIRE( fltcmp::eq(x(1), 3) );
}



TEMPLATE_TEST_CASE("indexing", "[mpimat]", float, double)
{
  len_t n = 2;
  
  cpumat<TestType> x_cpu = cpumat<TestType>(n, n);
  
  TestType *x_d = x_cpu.data_ptr();
  
  for (len_t i=0; i<n*n; i++)
    x_d[i] = (TestType) i+1;
  
  mpimat<TestType> x = mpihelpers::cpu2mpi(x_cpu, g, 1, 1);
  mpimat<TestType> y(g, n, n, 1, 1);
  
  y.fill_linspace(1, n*n);
  REQUIRE( (x == y) );
  
  y.fill_val(1.f);
  REQUIRE( (x != y) );
}



TEMPLATE_TEST_CASE("diag", "[cpumat]", float, double)
{
  len_t m = 4;
  len_t n = 3;
  
  mpimat<TestType> x(g, m, n, 1, 1);
  x.fill_linspace(1, m*n);
  
  cpuvec<TestType> v;
  x.diag(v);
  REQUIRE( fltcmp::eq(v(0), 1) );
  REQUIRE( fltcmp::eq(v(1), 6) );
  REQUIRE( fltcmp::eq(v(2), 11) );
  
  x.antidiag(v);
  REQUIRE( fltcmp::eq(v(0), 4) );
  REQUIRE( fltcmp::eq(v(1), 7) );
  REQUIRE( fltcmp::eq(v(2), 10) );
}



TEMPLATE_TEST_CASE("rev", "[cpumat]", float, double)
{
  len_t n = 2;
  
  mpimat<TestType> x(g, n, n, 1, 1);
  x.fill_linspace(1, n*n);
  
  x.rev_cols();
  REQUIRE( fltcmp::eq(x(0, 0), 3) );
  REQUIRE( fltcmp::eq(x(1, 0), 4) );
  
  // x.rev_rows();
  // REQUIRE( fltcmp::eq(x(0, 0), 4) );
  // REQUIRE( fltcmp::eq(x(1, 0), 3) );
}



TEMPLATE_TEST_CASE("get row/col", "[cpumat]", float, double)
{
  len_t n = 2;
  
  mpimat<TestType> x(g, n, n, 1, 1);
  x.fill_linspace(1, n*n);
  
  cpuvec<TestType> v;
  
  x.get_row(1, v);
  REQUIRE( fltcmp::eq(x(1, 0), v(0)) );
  REQUIRE( fltcmp::eq(x(1, 1), v(1)) );
  
  x.get_col(0, v);
  REQUIRE( fltcmp::eq(x(0, 0), v(0)) );
  REQUIRE( fltcmp::eq(x(1, 0), v(1)) );
}
