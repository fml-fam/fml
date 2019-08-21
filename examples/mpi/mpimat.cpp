#include <mpi/grid.hh>
#include <mpi/mpimat.hh>
#include <mpi/mpihelpers.hh>


int main()
{
  grid g = grid(PROC_GRID_SQUARE);
  g.print();
  
  mpimat<float> x = mpimat<float>(g, 5, 5, 2, 2);
  x.info();
  
  x.fill_eye();
  x.scale(3);
  
  x.print();
  
  g.exit();
  g.finalize();
  
  return 0;
}
