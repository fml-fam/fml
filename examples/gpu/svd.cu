#include <fml/gpu.hh>


int main()
{
  auto c = fml::gpuhelpers::new_card(0);
  
  len_t m = 3;
  len_t n = 2;
  
  fml::gpumat<float> x(c, m, n);
  x.fill_linspace(1, m*n);
  
  x.info();
  x.print(0);
  
  fml::gpuvec<float> s(c);
  fml::gpumat<float> u(c), vt(c);
  fml::linalg::svd(x, s, u, vt);
  
  s.info();
  s.print();
  
  u.info();
  u.print();
  
  return 0;
}
