#include <cpu/cpumat.hh>
#include <cpu/linalg.hh>


int main()
{
  len_t n = 2;
  
  cpumat<float> x = cpumat<float>(n, n);
  cpumat<float> y = cpumat<float>(n, n);
  
  float *x_d = x.data_ptr();
  float *y_d = y.data_ptr();
  
  for (len_t i=0; i<n*n; i++)
    x_d[i] = (float) i+1;
  
  for (len_t i=0; i<n*n; i++)
    y_d[i] = (float) (n*n)-i;
  
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
