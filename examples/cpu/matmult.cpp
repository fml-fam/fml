#include <fml/cpu.hh>


int main()
{
  len_t n = 2;
  
  fml::cpumat<float> x(n, n);
  fml::cpumat<float> y(n, n);
  
  x.fill_linspace(1, n*n);
  y.fill_linspace(n*n, 1);
  
  fml::cpumat<float> z = fml::linalg::matmult(false, false, 1.0f, x, y);
  z.print();
  
  fml::linalg::matmult(true, false, 1.0f, x, y, z);
  z.print();
  
  fml::linalg::matmult(false, true, 1.0f, x, y, z);
  z.print();
  
  fml::linalg::matmult(true, true, 1.0f, x, y, z);
  z.print();
  
  return 0;
}
