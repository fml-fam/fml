#include "../catch.hpp"

#include <mpi/grid.hh>
#include <mpi/mpihelpers.hh>
#include <mpi/mpimat.hh>

using namespace arraytools;

extern fml::grid g;


TEMPLATE_TEST_CASE("mpi2cpu_all", "[mpimat]", float, double)
{
  len_t m, n;
  int mb, nb;
  
  SECTION( "small blocks" )
  {
    m = 3, n = 2, mb = 1, nb = 1;
    
    fml::mpimat<TestType> dx(g, m, n, mb, nb);
    fml::cpumat<TestType> x_true(m, n);
    
    dx.fill_linspace(1, m*n);
    x_true.fill_linspace(1, m*n);
    fml::cpumat<TestType> x_test = fml::mpihelpers::mpi2cpu_all(dx);
    REQUIRE( x_test == x_true );
    
    dx.fill_eye();
    fml::mpihelpers::mpi2cpu_all(dx, x_test);
    x_true.fill_eye();
    REQUIRE( x_test == x_true );
  }
  
  SECTION( "larger blocks" )
  {
    m = 13, n = 11, mb = 5, nb = 3;
    
    fml::mpimat<TestType> dx(g, m, n, mb, nb);
    fml::cpumat<TestType> x_true(m, n);
    
    dx.fill_linspace(1, m*n);
    x_true.fill_linspace(1, m*n);
    fml::cpumat<TestType> x_test = fml::mpihelpers::mpi2cpu_all(dx);
    REQUIRE( x_test == x_true );
    
    dx.fill_eye();
    fml::mpihelpers::mpi2cpu_all(dx, x_test);
    x_true.fill_eye();
    REQUIRE( x_test == x_true );
  }
}



TEMPLATE_TEST_CASE("mpi2cpu", "[mpimat]", float, double)
{
  int check;
  len_t m, n;
  int mb, nb;
  
  SECTION( "small blocks" )
  {
    m = 3, n = 2, mb = 1, nb = 1;
    
    fml::mpimat<TestType> dx(g, m, n, mb, nb);
    fml::cpumat<TestType> x_test;
    
    dx.fill_linspace(1, m*n);
    
    if (g.rank0())
    {
      x_test.resize(m, n);
      x_test.fill_linspace(1, m*n);
    }
    
    fml::mpihelpers::mpi2cpu(dx, x_test);
    if (g.rank0())
    {
      fml::cpumat<TestType> x_true(m, n);
      x_true.fill_linspace(1, m*n);
      
      check = (int) (x_test == x_true);
    }
    else
      check = 0;
    
    g.allreduce(1, 1, &check);
    REQUIRE( check == 1 );
  }
  
  SECTION( "larger blocks" )
  {
    m = 13, n = 11, mb = 5, nb = 3;
    
    fml::mpimat<TestType> dx(g, m, n, mb, nb);
    fml::cpumat<TestType> x_test;
    
    dx.fill_linspace(1, m*n);
    
    if (g.rank0())
    {
      x_test.resize(m, n);
      x_test.fill_linspace(1, m*n);
    }
    
    fml::mpihelpers::mpi2cpu(dx, x_test);
    if (g.rank0())
    {
      fml::cpumat<TestType> x_true(m, n);
      x_true.fill_linspace(1, m*n);
      
      check = (int) (x_test == x_true);
    }
    else
      check = 0;
    
    g.allreduce(1, 1, &check);
    REQUIRE( check == 1 );
  }
}



TEMPLATE_TEST_CASE("cpu2mpi", "[mpimat]", float, double)
{
  len_t m, n;
  int mb, nb;
  
  SECTION( "small blocks" )
  {
    m = 3, n = 2, mb = 1, nb = 1;
    
    fml::mpimat<TestType> dx_test(g, mb, nb);
    fml::mpimat<TestType> dx_true(g, m, n, mb, nb);
    dx_true.fill_linspace(1, m*n);
    fml::cpumat<TestType> x(m, n);
    x.fill_linspace(1, m*n);
    
    fml::mpihelpers::cpu2mpi(x, dx_test);
    
    REQUIRE( dx_test == dx_true );
  }
  
  SECTION( "larger blocks" )
  {
    m = 13, n = 11, mb = 5, nb = 3;
    
    fml::mpimat<TestType> dx_test(g, mb, nb);
    fml::mpimat<TestType> dx_true(g, m, n, mb, nb);
    dx_true.fill_linspace(1, m*n);
    fml::cpumat<TestType> x(m, n);
    x.fill_linspace(1, m*n);
    
    fml::mpihelpers::cpu2mpi(x, dx_test);
    
    REQUIRE( dx_test == dx_true );
  }
}



TEMPLATE_TEST_CASE("mpi2mpi", "[mpimat]", float, double)
{
  len_t m, n;
  int mb, nb;
  
  SECTION( "small blocks" )
  {
    m = 3, n = 2, mb = 1, nb = 1;
    
    fml::mpimat<TestType> dx_test(g, mb, nb);
    fml::mpimat<TestType> dx_true(g, m, n, mb, nb);
    dx_true.fill_linspace(1, m*n);
    
    fml::mpihelpers::mpi2mpi(dx_true, dx_test);
    
    REQUIRE( dx_test == dx_true );
  }
  
  SECTION( "larger blocks" )
  {
    m = 13, n = 11, mb = 5, nb = 3;
    
    fml::mpimat<TestType> dx_test(g, mb, nb);
    fml::mpimat<TestType> dx_true(g, m, n, mb, nb);
    dx_true.fill_linspace(1, m*n);
    
    fml::mpihelpers::mpi2mpi(dx_true, dx_test);
    
    REQUIRE( dx_test == dx_true );
  }
}
