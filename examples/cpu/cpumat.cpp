#include <cpu/cpumat.hh>


int main()
{
  cpumat<float> x = cpumat<float>(5, 5);
  len_t m = x.nrows();
  len_t n = x.ncols();
  printf("%dx%d\n", m, n);
  
  x.fill_eye();
  x.print();
  
  x.scale(3);
  x.print();
  
  x.fill_runif(1234);
  x.print();
  
  return 0;
}
