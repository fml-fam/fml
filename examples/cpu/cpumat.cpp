#include <fml/cpu.hh>


int main()
{
  fml::cpumat<float> x(5, 5);
  x.info();
  
  x.fill_eye();
  x.scale(3);
  x.print();
  
  x.fill_runif(1234u);
  x.print();
  
  return 0;
}
