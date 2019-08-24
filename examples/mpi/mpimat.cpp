#include <mpi/grid.hh>
#include <mpi/mpimat.hh>
#include <mpi/mpihelpers.hh>


int main()
{
  grid g = grid(PROC_GRID_SQUARE);
  g.info();
  
  mpimat<float> x = mpimat<float>(g, 5, 5, 2, 2);
  x.info();
  
  x.fill_eye();
  x.scale(3);
  x.print();
  
  x.fill_runif(1234u);
  x.print();
  
  g.exit();
  g.finalize();
  
  return 0;
}
