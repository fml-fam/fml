#include <gpu/card.hh>
#include <gpu/gpumat.hh>

#include <cpu/cpumat.hh>
#include <gpu/gpuhelpers.hh>


int main()
{
  card c(0);
  c.info();
  
  double *d = (double*) c.mem_alloc(134217728 * sizeof(*d));
  c.info();
  c.mem_free(d);
  c.info();
  
  return 0;
}
