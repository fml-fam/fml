#include <mpi/mpimat.hh>
#include <mpi/linalg.hh>


int main()
{
  grid g = grid(PROC_GRID_SQUARE);
  g.info();
  
  len_t m = 3;
  len_t n = 2;
  
  mpimat<float> x(g, m, n, 1, 1);
  x.fill_linspace(1, m*n);
  
  x.info();
  x.print(0);
  
  cpuvec<float> s;
  linalg::svd(x, s);
  
  if (g.rank0())
  {
    s.info();
    s.print();
  }
  
  g.exit();
  g.finalize();
  
  return 0;
}
