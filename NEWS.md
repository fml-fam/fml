# NEWS

## Release 0.4-1 (11/15/2021):

New:
  * Added set_math_mode() method for card objects
  * Added pow() method for cpuvec and gpuvec objects
  * Added dimops::rowsweep() and dimops::colsweep()

API Changes:
  * None

Bug Fixes:
  * Fixed linalg::rsvd() issues with mpimat.

Internal Changes of Note:
  * None

Documentation:
  * None





## Release 0.4-0 (8/23/2020):

New:
  * Added linalg::dot()
  * Added linalg::trinv()
  * Added linalg::rsvd()
  * Added par utils get_device_count() and get_device_num()
  * Many parmat changes (still experimental).

API Changes:
  * linalg::matmult() can now accept mixed matrix/vector arguments.

Bug Fixes:
  * Changed gpumat crossprod() and tcrossprod() to use Xgemm() instead of
    Xsyrk() for better run-time performance.

Internal Changes of Note:
  * Internal linalg headers have been re-organized. This has no effect if
  you use the `linalg.hh` headers or the main type headers (`cpu.hh`, `gpu.hh`, `mpi.hh`).
  * Switched linalg::qr_Q() and linalg::lq_Q() to use `Xorgqr()`/`Xorglq()` except in gpumat lq_Q().
  * Changed `uplo` type in gpu_utils::lacpy() to char.

Documentation:
  * Fixed card synch() method documentation.





## Release 0.3-0 (6/20/2020):

New:
  * Added singular backend headers.
      - `#include <fml/cpu.hh>` includes all standard headers in `src/fml/cpu`.
      - `#include <fml/gpu.hh>` includes all standard headers in `src/fml/gpu`.
      - etc.
  * Added `fml::card_sp_t` typedef for `std::shared_ptr<fml::card>` in `card.hh`.
  * Added min/max methods for cpuvec and gpuvec.
  * Added to linalg namespace for cpumat, gpumat, and mpimat:
      - norm_1()
      - norm_I()
      - norm_F()
      - norm_M()
  * More OpenMP usage in CPU and MPI backends.

API Changes:
  * All headers are now contained in the `fml/` tree.
      - `src/cpu/cpumat.hh` becomes `src/fml/cpu/cpumat.hh`
      - `src/gpu/gpumat.hh` becomes `src/fml/gpu/gpumat.hh`
      - etc.
  * All classes and namespaces are now in the fml namespace.
      - `cpumat<float>` becomes `fml::cpumat<float>`
      - `linalg::svd()` becomes `fml::linalg::svd()`
      - etc.
  * All Xhelpers namespaces are renamed to the singular copy namespace.
      - `cpuhelpers::cpu2cpu()` becomes `fml::copy::cpu2cpu()`
      - `gpuhelpers::cpu2gpu()` becomes `fml::copy::cpu2gpu()`
      - etc.
  * Rename linalg::tssvd() to linalg::qrsvd(), and added LQ case when m<n.
  * Added support for m<n case (via a transpose) of gpumat linalg::svd().

Bug Fixes:
  * Fixed an indexing bug in mpimat linalg::cpsvd() when m>n.

Documentation:
  * Fixed some formatting issues in generated documentation.





## Release 0.2-1 (5/28/20):

New: None

API Changes: None

Bug Fixes:
  * Fixed det() bug for cpumat.
  * Expanded cpu2mpi() to accept objects of different fundamental types.
  * Fixed indexing bug in mpi2cpu().
  * Fixed mixed type bug in mpi2cpu().
  * Fixed 0 value ownership issue with mpimat.
  * Fixed 0 value ownership issue with invert() for mpimat.

Documentation: None





## Release 0.2-0 (5/20/20):

New:
  * Added to linalg namespace for cpumat, gpumat, and mpimat:
      - qr()
      - qr_Q()
      - qr_R()
      - lq()
      - lq_L()
      - lq_Q()
      - tssvd()
      - cpsvd()
      - det()
      - chol()
  * Created dimops namespace
  * Added to dimops namespace for cpumat, gpumat, and mpimat:
      - rowsums()
      - rowmeans()
      - colsums()
      - colmeans()
      - scale()
  * Created stats namespace
  * Added to stats namespace for cpumat, gpumat, and mpimat:
      - pca()

API Changes:
  * Added default argument to gpuhelpers::new_card()

Bug Fixes:
  * Fixed major resize bug in mpimat not updating descriptor arrays.
  * Changed matrix object `==` tests to use `arraytools::fltcmp::eq()` instead
    of `==` on fundamental types.
  * Changed seed setter behavior in mpimat rng functions: local seed is now
    computed as seed + myrow + nprow*mycol.
  * Fixed several memory issues in gpumat linalg::svd().
  * Fixed an internal logic error in gpumat linalg::svd().

Documentation:
  * Minor corrections; no major changes.





## Release 0.1-0 (2/5/2020):

New:
  * Created cpumat and cpuvec classes
  * Created gpumat and gpuvec classes
  * Created mpimat class
  * Created parmat_cpu and parmat_gpu classes
  * Created linalg namespace
  * Added to linalg namespace for cpumat, gpumat, and mpimat:
      - add()
      - matmult()
      - crossprod()
      - tcrossprod()
      - xpose()
      - lu()
      - svd()
      - eigen_sym()
      - invert()
      - solve()
  * Added to linalg namespace for parmat:
      - crossprod()
  * Created cpuhelpers namespace
  * Added to cpuhelpers namespace
      - cpu2cpu()
  * Created gpuhelpers namespace
  * Added to gpuhelpers namespace
      - gpu2cpu()
      - cpu2gpu()
      - gpu2gpu()
  * Created mpihelpers namespace
  * Added to mpihelpers namespace
      - mpi2cpu()
      - cpu2mpi()
      - mpi2mpi()

API Changes: None

Bug Fixes: None

Documentation:
  * Added documentation for most things.
