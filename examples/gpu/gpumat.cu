#include <gpu/card.hh>
#include <gpu/gpumat.hh>

#include <cpu/cpumat.hh>
#include <gpu/gpuhelpers.hh>


int main()
{
  auto c = std::make_shared<card>(0);
  c->info();
  
  len_t n = 5;
  gpumat<float> x(c, n, n);
  x.info();
  
  x.fill_eye();
  x.scale(3.f);
  x.print();
  
  x.fill_linspace(1.f, (float) n*n);
  
  cpumat<float> x_cpu = gpuhelpers::gpu2cpu(x);
  x_cpu.print();
  
  return 0;
}
