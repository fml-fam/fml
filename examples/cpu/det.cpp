#include <cpu/cpumat.hh>
#include <cpu/linalg.hh>


static inline void print_det(int sign, float modulus)
{
  printf("sgn = %d\n", sign);
  printf("mod = %f\n", modulus);
  printf("sgn*exp(mod) = %f\n", sign*exp(modulus));
  printf("\n");
}



int main()
{
  len_t n = 2;
  
  cpumat<float> x = cpumat<float>(n, n);
  float *x_d = x.data_ptr();
  
  for (len_t i=0; i<n*n; i++)
    x_d[i] = (float) i+1;
  
  x.info();
  x.print(0);
  
  int sign;
  float modulus;
  linalg::det(x, sign, modulus);
  print_det(sign, modulus);
  
  n = 4;
  x.resize(n, n);
  x_d = x.data_ptr();
  
  for (len_t i=0; i<n*n; i++)
    x_d[i] = (float) i+1;
  
  x.info();
  x.print(0);
  
  linalg::det(x, sign, modulus);
  print_det(sign, modulus);
  
  return 0;
}
