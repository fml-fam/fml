#ifndef FML_UNIVEC_H
#define FML_UNIVEC_H


#include "types.hh"

template <typename T>
class univec
{
  public:
    len_t size() const {return _size;};
    T* data_ptr() {return data;};
    T* data_ptr() const {return data;};
  
  protected:
    len_t _size;
    T *data;
    bool free_data;
    bool should_free() const {return free_data;};
};


#endif
