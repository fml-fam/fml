#include <cpu/cpumat.hh>
#include <cpu/linalg.hh>


int main()
{
  len_t m = 3;
  len_t n = 2;
  
  fml::cpumat<float> x(m, n);
  x.fill_linspace(1, m*n);
  
  x.info();
  x.print(0);
  
  fml::cpuvec<float> s;
  fml::cpumat<float> u, vt;
  fml::linalg::svd(x, s, u, vt);
  
  s.info();
  s.print();
  
  u.info();
  u.print();
  
  return 0;
}
