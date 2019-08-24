#include <cpu/cpumat.hh>


int main()
{
  cpumat<float> x = cpumat<float>(5, 5);
  x.info();
  
  x.fill_eye();
  x.scale(3);
  x.print();
  
  x.fill_runif(1234u);
  x.print();
  
  return 0;
}
