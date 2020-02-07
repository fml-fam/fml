Release 0.2-0 (//):

New:

API Changes: None

Bug Fixes:

Documentation:




Release 0.1-0 (2/5/2020):

New:
  * created cpumat and cpuvec classes
  * created gpumat and gpuvec classes
  * created mpimat class
  * created parmat_cpu and parmat_gpu classes
  * added to linalg namespace for cpumat, gpumat, and mpimat:
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
  * added to linalg namespace for parmat:
      - crossprod()
  * added to cpuhelpers namespace
      - cpu2cpu()
  * added to gpuhelpers namespace
      - gpu2cpu()
      - cpu2gpu()
      - gpu2gpu()
  * added to mpihelpers namespace
      - mpi2cpu()
      - cpu2mpi()
      - mpi2mpi()

API Changes: None

Bug Fixes: None

Documentation:
  * Added documentation for most things.
