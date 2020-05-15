#include <cpu/cpumat.hh>
#include <cpu/linalg.hh>


int main()
{
  len_t n = 2;
  
  cpumat<float> x(n, n);
  cpumat<float> y(n, n);
  
  x.fill_linspace(1, n*n);
  y.fill_linspace(n*n, 1);
  
  cpumat<float> z = linalg::matmult(false, false, 1.0f, x, y);
  z.print();
  
  linalg::matmult(true, false, 1.0f, x, y, z);
  z.print();
  
  linalg::matmult(false, true, 1.0f, x, y, z);
  z.print();
  
  linalg::matmult(true, true, 1.0f, x, y, z);
  z.print();
  
  return 0;
}
