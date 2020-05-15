#include <gpu/card.hh>
#include <gpu/gpuhelpers.hh>
#include <gpu/gpumat.hh>
#include <gpu/linalg.hh>


int main()
{
  auto c = gpuhelpers::new_card(0);
  
  len_t m = 3;
  len_t n = 2;
  
  gpumat<float> x(c, m, n);
  x.fill_linspace(1, m*n);
  
  x.info();
  x.print(0);
  
  gpuvec<float> s(c);
  gpumat<float> u(c), vt(c);
  linalg::svd(x, s, u, vt);
  
  s.info();
  s.print();
  
  u.info();
  u.print();
  
  return 0;
}
