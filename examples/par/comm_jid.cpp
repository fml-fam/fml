#include <fml/par/comm.hh>


int main()
{
  fml::comm c = fml::comm();
  
  c.info();
  c.barrier();
  
  auto jid = c.jid(7);
  for (int i=0; i<c.size(); i++)
  {
    if (c.rank() == i)
    {
      printf("rank %d: ", c.rank());
      for (int i=0; i<jid.size(); i++)
        printf("%d ", jid[i]);
      
      printf("\n");
    }
    
    c.barrier();
  }
  
  c.finalize();
  
  return 0;
}
