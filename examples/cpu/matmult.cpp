#include <cpu/cpumat.hh>
#include <cpu/linalg.hh>


int main()
{
  len_t n = 2;
  
  cpumat<float> x = cpumat<float>(n, n);
  cpumat<float> y = cpumat<float>(n, n);
  
  x.fill_linspace(1.f, (float) n*n);
  y.fill_linspace((float) n*n, 1.f);
  
  cpumat<float> z = linalg::matmult(false, false, 1.0f, x, y);
  z.print();
  
  linalg::matmult_noalloc(true, false, 1.0f, x, y, z);
  z.print();
  
  linalg::matmult_noalloc(false, true, 1.0f, x, y, z);
  z.print();
  
  linalg::matmult_noalloc(true, true, 1.0f, x, y, z);
  z.print();
  
  return 0;
}
