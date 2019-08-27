#include <cpu/cpumat.hh>
#include <cpu/linalg.hh>


int main()
{
  len_t m = 3;
  len_t n = 2;
  
  cpumat<float> x = cpumat<float>(m, n);
  x.fill_linspace(1.f, (float)m*n);
  
  x.info();
  x.print(0);
  
  cpuvec<float> s;
  cpumat<float> u, vt;
  linalg::svd(x, s, u, vt);
  
  s.info();
  s.print();
  
  u.info();
  u.print();
  
  return 0;
}
