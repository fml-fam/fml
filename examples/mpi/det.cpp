#include <mpi/mpimat.hh>
#include <mpi/linalg.hh>


static inline void print_det(int sign, float modulus)
{
  printf("sgn = %d\n", sign);
  printf("mod = %f\n", modulus);
  printf("sgn*exp(mod) = %f\n", sign*exp(modulus));
  printf("\n");
}



int main()
{
  grid g = grid(PROC_GRID_SQUARE);
  g.info();
  
  len_t n = 2;
  mpimat<float> x(g, n, n, 1, 1);
  x.fill_linspace(1, n*n);
  
  x.info();
  x.print(0);
  
  int sign;
  float modulus;
  linalg::det(x, sign, modulus);
  if (g.rank0())
    print_det(sign, modulus);
  
  n = 4;
  x.resize(n, n);
  x.fill_linspace(1, n*n);
  
  x.info();
  x.print(0);
  
  linalg::det(x, sign, modulus);
  if (g.rank0())
    print_det(sign, modulus);
  
  g.finalize();
  
  return 0;
}
