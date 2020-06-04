#include <fml/mpi.hh>


int main()
{
  fml::grid g = fml::grid(fml::PROC_GRID_SQUARE);
  g.info();
  
  fml::mpimat<float> x(g, 5, 5, 2, 2);
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
