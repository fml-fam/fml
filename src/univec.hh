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
    void printval(const T val, uint8_t ndigits) const;
};



template <>
inline void univec<int>::printval(const int val, uint8_t ndigits) const
{
  (void)ndigits;
  printf("%d ", val);
}

template <typename T>
void univec<T>::printval(const T val, uint8_t ndigits) const
{
  printf("%.*f ", ndigits, val);
}


#endif
