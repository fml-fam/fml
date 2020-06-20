#include <fml/par/comm.hh>


int main()
{
  fml::comm c = fml::comm();
  
  c.info();
  c.barrier();
  
  auto jid = c.jid(7);
  for (int rank=0; rank<c.size(); rank++)
  {
    if (c.rank() == rank)
    {
      printf("rank %d: ", c.rank());
      for (size_t i=0; i<jid.size(); i++)
        printf("%d ", jid[i]);
      
      printf("\n");
    }
    
    c.barrier();
  }
  
  c.finalize();
  
  return 0;
}
