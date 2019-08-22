#include "../catch.hpp"
#include "../fltcmp.hh"

#include <mpi/bcutils.hh>
#include <mpi/grid.hh>
#include <mpi/mpimat.hh>


extern grid g;


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
  else
    testval = 0.f;
  
  g.allreduce(1, 1, &testval, 'A');
  REQUIRE( fltcmp::eq(testval, 1) );
  
  mpimat<TestType> y(g);
  y.set(g, data, m, n, mb, nb);
  y.fill_zero();
  y.~mpimat();
  
  if (g.rank0())
    testval = data[0];
  else
    testval = 0.f;
  
  g.allreduce(1, 1, &testval, 'A');
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



TEMPLATE_TEST_CASE("scale", "[cpumat]", float, double)
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
