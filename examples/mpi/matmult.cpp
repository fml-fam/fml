#include <mpi/grid.hh>
#include <mpi/linalg.hh>
#include <mpi/mpimat.hh>


int main()
{
  grid g = grid(PROC_GRID_SQUARE);
  g.info();
  
  len_t n = 2;
  
  mpimat<float> x(g, n, n, 1, 1);
  mpimat<float> y(g, n, n, 1, 1);
  x.fill_linspace(1, n*n);
  y.fill_linspace(n*n, 1);
  
  mpimat<float> z = linalg::matmult(false, false, 1.0f, x, y);
  z.info();
  z.print();
  
  linalg::matmult(true, false, 1.0f, x, y, z);
  z.print();
  
  linalg::matmult(false, true, 1.0f, x, y, z);
  z.print();
  
  linalg::matmult(true, true, 1.0f, x, y, z);
  z.print();
  
  g.exit();
  g.finalize();
  
  return 0;
}
