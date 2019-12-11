// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_INTERNALS_GPUSCALAR_H
#define FML_GPU_INTERNALS_GPUSCALAR_H
#pragma once


#include "../card.hh"


template <typename T>
class gpuscalar
{
  public:
    gpuscalar(std::shared_ptr<card> gpu);
    gpuscalar(std::shared_ptr<card> gpu, const T v);
    ~gpuscalar();
    
    void set_zero();
    void set_val(const T v);
    void get_val(T *v);
    
    T* data_ptr() {return data;};
    T* data_ptr() const {return data;};
  
  protected:
    std::shared_ptr<card> c;
    T *data;
};



template <typename T>
gpuscalar<T>::gpuscalar(std::shared_ptr<card> gpu)
{
  c = gpu;
  data = (T*) c->mem_alloc(sizeof(T));
}



template <typename T>
gpuscalar<T>::gpuscalar(std::shared_ptr<card> gpu, const T v)
{
  c = gpu;
  data = (T*) c->mem_alloc(sizeof(T));
  c->mem_cpu2gpu(data, &v, sizeof(T));
}



template <typename T>
gpuscalar<T>::~gpuscalar()
{
  c->mem_free(data);
  data = NULL;
}



template <typename T>
void gpuscalar<T>::set_zero()
{
  c->mem_set(data, 0, sizeof(T));
}



template <typename T>
void gpuscalar<T>::set_val(const T v)
{
  c->mem_set(data, &v, sizeof(T));
}



template <typename T>
void gpuscalar<T>::get_val(T *v)
{
  c->mem_gpu2cpu(v, data, sizeof(T));
}


#endif
