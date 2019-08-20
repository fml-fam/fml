#include <mpi/grid.hh>
#include <mpi/mpimat.hh>
#include <mpi/mpihelpers.hh>


int main()
{
  grid g = grid(PROC_GRID_SQUARE);
  g.print();
  
  mpimat<float> x = mpimat<float>(g, 5, 5, 2, 2);
  
  len_t m = x.nrows();
  len_t n = x.ncols();
  g.printf(0, 0, "%dx%d\n", m, n);
  
  x.fill_eye();
  x.scale(3);
  
  cpumat<float> x_gbl = mpihelpers::mpi2cpu(x);
  if (g.rank0())
    x_gbl.print();
  
  g.exit();
  g.finalize();
  
  return 0;
}
