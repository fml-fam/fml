# NEWS

## Release 0.2-1 (//):

New: None

API Changes: None

Bug Fixes:
  * Fixed det() bug for cpumat.
  * Expanded cpu2mpi() to accept objects of different fundamental types.
  * Fixed indexing bug in mpi2cpu().
  * Fixed mixed type bug in mpi2cpu().

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
