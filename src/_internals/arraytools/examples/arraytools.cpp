#include <arraytools.hpp>


int main()
{
  double *a, *b;
  arraytools::alloc(3, &a);
  arraytools::alloc(3, &b);
  
  try {
    arraytools::check_allocs(a, b);
    printf("allocations ok\n");
  }
  catch (const std::bad_alloc& e) {
    printf("caught bad alloc\n");
  }
  
  
  
  double *c = NULL;
  
  try {
    arraytools::check_allocs(a, b, c);
    printf("allocations ok\n");
  }
  catch (const std::bad_alloc& e) {
    printf("caught bad alloc\n");
  }
  
  return 0;
}
