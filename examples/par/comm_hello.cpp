#include <par/comm.hh>


int main()
{
  comm c = comm();
  
  c.info();
  c.barrier();
  
  for (int i=0; i<c.size(); i++)
  {
    c.printf(i, "Hello from rank %d of %d\n", c.rank(), c.size());
    c.barrier();
  }
  
  c.finalize();
  
  return 0;
}
