#include <bench.hh>
#include <memuse.hh>

#include <mpi/grid.hh>
#include <mpi/linalg.hh>
#include <mpi/mpihelpers.hh>
#include <mpi/mpimat.hh>


typedef float REAL;


int main()
{
  len_t N = 1000;
  int NB = 16;
  int KMIN = 2;
  int KMAX = 10;
  
  
  grid g = grid(PROC_GRID_SQUARE);
  g.info();
  
  mpimat<REAL> x(g, N, N, NB, NB);
  mpimat<REAL> tmp(g, N, N, NB, NB);
  mpimat<REAL> p(g, N, N, NB, NB);
  x.fill_rnorm(1234, 0, 1);
  
  memuse m;
  m.howbig(3*N, N);
  
  bench b;
  if (g.rank0())
    b.print_header("GEMM: %dx%d * %dx%d (%s)", N, N, N, N, m.print_str().c_str());
  
  for (int k=KMIN; k<KMAX; k++)
  {
    mpihelpers::mpi2mpi(x, p);
    
    b.start();
    for (int i=1; i<k; i++)
    {
      mpihelpers::mpi2mpi(p, tmp);
      linalg::matmult(false, false, 1.0f, x, tmp, p);
    }
    
    b.stop();
    if (g.rank0())
      b.report("k=" + std::to_string(k));
  }
  
  g.exit();
  g.finalize();
  
  return 0;
}
