#include <gpu/card.hh>
#include <gpu/gpumat.hh>
#include <gpu/linalg.hh>

#include <gpu/gpuhelpers.hh>


int main()
{
  auto c = gpuhelpers::new_card(0);
  c->info();
  
  len_t n = 5;
  gpumat<__half> x(c, n, n);
  x.fill_zero();
  
  gpumat<__half> y(c, n, n);
  y.fill_zero();
  
  gpumat<__half> z = linalg::matmult(false, false, (__half)1.f, x, y);
  z.info();
  
  return 0;
}
