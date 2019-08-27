#ifndef FML_UNIVEC_H
#define FML_UNIVEC_H


#include "types.hh"

template <typename T>
class univec
{
  public:
    len_t size() const {return len;};
    T* data_ptr() {return data;};
    T* data_ptr() const {return data;};
  
  protected:
    len_t len;
    T *data;
};


#endif
