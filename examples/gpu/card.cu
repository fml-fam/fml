#include <gpu/card.hh>


int main()
{
  fml::card c(0);
  c.info();
  
  double *d = (double*) c.mem_alloc(134217728 * sizeof(*d));
  c.info();
  c.mem_free(d);
  c.info();
  
  return 0;
}
