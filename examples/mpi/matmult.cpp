#include <mpi/grid.hh>
#include <mpi/linalg.hh>
#include <mpi/mpimat.hh>
#include <mpi/mpihelpers.hh>


int main()
{
  grid g = grid(PROC_GRID_SQUARE);
  g.print();
  
  len_t n = 2;
  cpumat<float> x_cpu = cpumat<float>(n, n);
  cpumat<float> y_cpu = cpumat<float>(n, n);
  
  float *x_d = x_cpu.data_ptr();
  float *y_d = y_cpu.data_ptr();
  
  for (len_t i=0; i<n*n; i++)
  {
    x_d[i] = (float) i+1;
    y_d[i] = (float) (n*n)-i;
  }
  
  mpimat<float> x = mpihelpers::cpu2mpi(x_cpu, g, 1, 1);
  mpimat<float> y = mpihelpers::cpu2mpi(y_cpu, g, 1, 1);
  
  mpimat<float> z = linalg::matmult(false, false, 1.0f, x, y);
  
  cpumat<float> z_gbl = mpihelpers::mpi2cpu(z);
  if (g.rank0())
    z_gbl.print();
  
  linalg::matmult_noalloc(true, false, 1.0f, x, y, z);
  mpihelpers::mpi2cpu_noalloc(z, z_gbl);
  if (g.rank0())
    z_gbl.print();
  
  linalg::matmult_noalloc(false, true, 1.0f, x, y, z);
  mpihelpers::mpi2cpu_noalloc(z, z_gbl);
  if (g.rank0())
    z_gbl.print();
  
  linalg::matmult_noalloc(true, true, 1.0f, x, y, z);
  mpihelpers::mpi2cpu_noalloc(z, z_gbl);
  if (g.rank0())
    z_gbl.print();
  
  g.exit();
  g.finalize();
  
  return 0;
}
