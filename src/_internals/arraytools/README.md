# arraytools

* **Version:** 0.1-0
* **Status:** [![Build Status](https://travis-ci.org/wrathematics/arraytools.png)](https://travis-ci.org/wrathematics/arraytools)
* **License:** [BSL-1.0](http://opensource.org/licenses/BSL-1.0)
* **Project home**: https://github.com/wrathematics/arraytools
* **Bug reports**: https://github.com/wrathematics/arraytools/issues


arraytools is a small, single file, header-only C++ 17 library that makes some minor improvements to working with dynamic arrays. It's very simple and permissively licensed, but you could probably re-create this yourself pretty easily. I'm only putting this here because I am using it across multiple projects. That said, if this looks interesting to you but there's some feature you want and it isn't hilariously out of scope (e.g. "re-implement std::vector"), then feel free to ask and I'll probably do it.

The library solves two major annoyances I have with dynamic arrays in C++. For one, unlike in C you have to cast the return of `malloc()`. This is stupid and annoying. Allocation functions in arraytools avoid this by accepting a reference. Also, guarding against a bad `malloc()` is usually verbose and complicated. We handle this with `check_allocs()`, which can detect an allocation failure and then free all passed arrays before throwing `std::bad_alloc()`. The function uses variadic templates, so it's annoying to write but easy to use.

There are some minor improvements as well. First, all length arguments are "number of elements", as opposed to number of bytes. Next, the `malloc()` and `calloc()` equivalents (`arraytools::alloc()` and `arraytools::zero_alloc()`) have a consistent API. Also the `realloc()` wrapper `arraytools::realloc()` automatically handles allocation failure in a simple way. Finally, the `free()` wrapper `arraytools::free()` will only call `std::free()` on a non-`NULL` pointer.



## Dependencies and Tests

There are no external dependencies. Tests use [catch2](https://github.com/catchorg/Catch2), a copy of which is included under `tests/catch`.

To build the tests, modify `tests/make.inc` as appropriate and type `make`.



## API

```C++
/**
 * Allocate an array. Wrapper around malloc().
 * 
 * @param[in] len Number of elements (not the number of bytes!).
 * @param[out] x Array to be allocated.
 */
template <typename T>
void alloc(const size_t len, T **x)

/**
 * Zero-allocate an array. Wrapper around calloc().
 * 
 * @param[in] len Number of elements (not the number of bytes!).
 * @param[out] x Array to be allocated.
 */
template <typename T>
void zero_alloc(const size_t len, T **x)

/**
 * Re-allocate an array. Wrapper around realloc(). If the realloc fails, the
 * pointer will be set to NULL.
 * 
 * @param[in] len Number of elements (not the number of bytes!).
 * @param[out] x Array to be re-allocated.
 */
template <typename T>
void realloc(const size_t len, T **x)

/**
 * Free an array if supplied pointer is not NULL. Wrapper around free().
 * 
 * @param[in] x Array to be allocated.
 */
template <typename T>
void free(T *x)

/** 
 * Copy one array onto another. Array types can differ. If they are the same, it
 * reduces to a memcpy() call.
 * 
 * @param[in] len Number of elements (not the number of bytes!).
 * @param[in] src Source array.
 * @param[out] dst Destination array.
 */
template <typename SRC, typename DST>
void copy(const size_t len, const SRC *src, DST *dst)

/**
 * Set an array's values to 0. Wrapper around memset().
 * 
 * @param[in] len Number of elements (not the number of bytes!).
 * @param[inout] x Array to be zeroed.
 */
template <typename T>
void zero(const size_t len, T *x)

/**
 * Check variable number of arrays. If one is NULL, then all others will be
 * automatically freed and std::bad_alloc() will be thrown.
 * 
 * @param[in] x Array.
 * @param[in] vax Optional more arrays.
 */
template <typename T, typename... VAT>
void check_allocs(T *x, VAT... vax)
```



## Examples

There are some examples in `examples/`, and more example usage in `tests/`. But for the sake of readme-completion, here's a simple example:

```c++
int *a, *b;
arraytools::alloc(2, &a);
arraytools::alloc(2, &b);

arraytools::check_allocs(a);
arraytools::check_allocs(a, b);

arraytools::zero(2, a);


// Will automatically free a and b and then throw std::badalloc()
TestType *c = NULL;
arraytools::check_allocs(a, b, c);
```
