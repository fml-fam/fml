#include "../catch.hpp"

#include <_internals/arraytools/src/arraytools.hpp>
#include <mpi/mpimat.hh>
#include <mpi/internals/mpi_utils.hh>

using namespace arraytools;

extern fml::grid g;


TEMPLATE_TEST_CASE("tri2zero - tall", "[mpi_utils]", float, double)
{
  len_t m = 5;
  len_t n = 3;
  int mb = 1;
  int nb = 1;
  fml::mpimat<TestType> x(g, m, n, mb, nb);
  fml::mpimat<TestType> truth(g, m, n, mb, nb);
  truth.fill_linspace(1, m*n);
  
  x.fill_linspace(1, m*n);
  fml::mpi_utils::tri2zero('U', false, g, m, n, x.data_ptr(), x.desc_ptr());
  truth.set(0, 1, 0);
  truth.set(0, 2, 0);
  truth.set(1, 2, 0);
  REQUIRE( x == truth );
  
  x.fill_linspace(1, m*n);
  fml::mpi_utils::tri2zero('U', true, g, m, n, x.data_ptr(), x.desc_ptr());
  truth.set(0, 0, 0);
  truth.set(1, 1, 0);
  truth.set(2, 2, 0);
  REQUIRE( x == truth );
  
  truth.fill_linspace(1, m*n);
  
  x.fill_linspace(1, m*n);
  fml::mpi_utils::tri2zero('L', false, g, n, n, x.data_ptr(), x.desc_ptr());
  truth.set(1, 0, 0);
  truth.set(2, 0, 0);
  truth.set(2, 1, 0);
  REQUIRE( x == truth );
  
  x.fill_linspace(1, m*n);
  fml::mpi_utils::tri2zero('L', false, g, m, n, x.data_ptr(), x.desc_ptr());
  truth.set(3, 0, 0);
  truth.set(4, 0, 0);
  truth.set(3, 1, 0);
  truth.set(4, 1, 0);
  truth.set(3, 2, 0);
  truth.set(4, 2, 0);
  REQUIRE( x == truth );
  
  x.fill_linspace(1, m*n);
  fml::mpi_utils::tri2zero('L', true, g, m, n, x.data_ptr(), x.desc_ptr());
  truth.set(0, 0, 0);
  truth.set(1, 1, 0);
  truth.set(2, 2, 0);
  REQUIRE( x == truth );
}



TEMPLATE_TEST_CASE("tri2zero - wide", "[mpi_utils]", float, double)
{
  len_t m = 3;
  len_t n = 5;
  int mb = 1;
  int nb = 1;
  fml::mpimat<TestType> x(g, m, n, mb, nb);
  fml::mpimat<TestType> truth(g, m, n, 1, 1);
  truth.fill_linspace(1, m*n);
  
  x.fill_linspace(1, m*n);
  fml::mpi_utils::tri2zero('L', false, g, m, n, x.data_ptr(), x.desc_ptr());
  truth.set(1, 0, 0);
  truth.set(2, 0, 0);
  truth.set(2, 1, 0);
  REQUIRE( x == truth );
  
  x.fill_linspace(1, m*n);
  fml::mpi_utils::tri2zero('L', true, g, m, n, x.data_ptr(), x.desc_ptr());
  truth.set(0, 0, 0);
  truth.set(1, 1, 0);
  truth.set(2, 2, 0);
  REQUIRE( x == truth );
  
  truth.fill_linspace(1, m*n);
  
  x.fill_linspace(1, m*n);
  fml::mpi_utils::tri2zero('U', false, g, m, m, x.data_ptr(), x.desc_ptr());
  truth.set(0, 1, 0);
  truth.set(0, 2, 0);
  truth.set(1, 2, 0);
  REQUIRE( x == truth );
  
  x.fill_linspace(1, m*n);
  fml::mpi_utils::tri2zero('U', false, g, m, n, x.data_ptr(), x.desc_ptr());
  truth.set(0, 3, 0);
  truth.set(0, 4, 0);
  truth.set(1, 3, 0);
  truth.set(1, 4, 0);
  truth.set(2, 3, 0);
  truth.set(2, 4, 0);
  REQUIRE( x == truth );
  
  x.fill_linspace(1, m*n);
  fml::mpi_utils::tri2zero('U', true, g, m, n, x.data_ptr(), x.desc_ptr());
  truth.set(0, 0, 0);
  truth.set(1, 1, 0);
  truth.set(2, 2, 0);
  REQUIRE( x == truth );
}
