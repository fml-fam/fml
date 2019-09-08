# fml

* **Version:** 0.1-0
* **License:** [BSL-1.0](http://opensource.org/licenses/BSL-1.0)
* **Project home**: https://github.com/wrathematics/fml
* **Bug reports**: https://github.com/wrathematics/fml/issues
* **Author:** Drew Schmidt


<img align="right" src="./docs/logo/fml_med.png" />

fml is the Fused Matrix Library, a multi-source, header-only C++ library for matrix computing.

The library provides 4 main classes: `cpumat`, `gpumat`, `parmat`, and `mpimat`. These are mostly what they sound like, but the particular details are:

* <font color="blue">CPU</font>: Single node cpu computing (multi-threaded if using multi-threaded BLAS and linking with OpenMP).
* <font color="green">GPU</font>: Single gpu computing via CUDA.
* <font color="orange">PAR</font>: Multi-node and/or multi-gpu computing via MPI and/or CUDA.
* <font color="red">MPI</font>: Multi-node computing via ScaLAPACK (+gpus if using [SLATE](http://icl.utk.edu/slate/)).

There are some differences in how objects of any particular type are constructed. But the high level APIs are largely the same between the objects. The goal is to be able to quickly create laptop-scale prototypes that are then easily converted into large scale gpu/multi-node/multi-gpu/multi-node+multi-gpu codes.



## Dependencies Other Software

There are no external header dependencies, but there are some shared libraries you need to have (more information below):

* CPU code needs [LAPACK](http://performance.netlib.org/lapack/) (I recommend [OpenBLAS](https://github.com/xianyi/OpenBLAS))
* GPU code needs [CUDA](https://developer.nvidia.com/cuda-downloads)
* MPI code needs [ScaLAPACK](http://performance.netlib.org/scalapack/)

Other software we use:

* Tests use [catch2](https://github.com/catchorg/Catch2) (a copy of which is included under `tests/`).

You can find some examples of how to use the library in the `examples/` tree. Right now there is no real build system beyond some ad hoc makefiles; but ad hoc is better than no hoc.

Depending on which class(es) you want to use, here are some general guidelines for using the library in your own project:

* CPU: `cpumat`
    - Compile with your favorite C++ compiler.
    - Link with LAPACK and BLAS (and ideally with OpenMP).
* GPU: `gpumat`
    - Compile with `nvcc`.
    - For most functionality, link with libcudart, libcublas, and libcusolver.  Link with libnvidia-ml if using nvml (if you're only using this, then you don't need `nvcc`; an ordinary C++ compiler will do).
* PAR: `parmat`
    - Compile with `mpicxx`.
    - TODO
* MPI: `mpimat`
    - Compile with `mpicxx`.
    - Link with libscalapack.

Check the makefiles in the `examples/` tree if none of that makes sense.



## Example

```C++
#include <cpu/cpumat.hh>
#include <cpu/linalg.hh>


int main()
{
  len_t m = 3;
  len_t n = 2;
  
  cpumat<float> x(m, n);
  x.fill_linspace(1.f, (float)m*n);
  
  x.info();
  x.print(0);
  
  cpuvec<float> s;
  linalg::svd(x, s);
  
  s.info();
  s.print();
  
  return 0;
}
```

Save as `svd.cpp` and build with:

```bash
g++ -I/path/to/fml/src -fopenmp svd.cpp -o svd -llapack -lblas
```

You should see output like

```
# cpumat 3x2 type=f
1 4 
2 5 
3 6 

# cpuvec 2 type=f
9.5080 0.7729 
```

The API is largely the same if we change the object storage, but we have to change the object initialization. For example, if `x` is an object of class `mpimat`, we still call `linalg::svd(x, s)`. The differences lie in the creation of the objects. Here is how we might change the above example to use distributed data:

```C++
#include <mpi/mpimat.hh>
#include <mpi/linalg.hh>


int main()
{
  grid g = grid(PROC_GRID_SQUARE);
  g.info();
  
  len_t m = 3;
  len_t n = 2;
  
  mpimat<float> x(g, m, n, 1, 1);
  x.fill_linspace(1.f, (float)m*n);
  
  x.info();
  x.print(0);
  
  cpuvec<float> s;
  linalg::svd(x, s);
  
  if (g.rank0())
  {
    s.info();
    s.print();
  }
  
  g.exit();
  g.finalize();
  
  return 0;
}
```

We can build it via:

```bash
mpicxx -I/path/to/fml/src svd.cpp -fopenmp  svd.cpp -o svd -lscalapack-openmpi
```

We can launch the example with multiple processes via

```bash
mpirun -np 4 ./svd
```

And here we see:

```
## Grid 0 2x2

# mpimat 3x2 on 2x2 grid type=f
1 4 
2 5 
3 6 

# cpuvec 2 type=f
9.5080 0.7729 
```



## Philosophy and Similar Projects

Some similar C/C++ projects worth mentioning:

* [Armadillo](http://arma.sourceforge.net/)
* [Eigen](http://eigen.tuxfamily.org/)
* [Boost](http://www.boost.org/)
* [PETSc](https://www.mcs.anl.gov/petsc/)
* [GSL](https://www.gnu.org/software/gsl/)

These are all great libraries which have stood the test of time and have many happy users. Armadillo in particular is worthy of a look, as it has a very nice interface and very extensive set of functions. However, to my knowledge, all of these focus exclusively on CPU computing. You can use things like [nvblas](https://docs.nvidia.com/cuda/nvblas/index.html) to offload some computations to GPU (ones that are very `gemm` heavy can do well), but this doesn't always achieve good performance, and it doesn't include distributed computing.

There are probably many other C++ frameworks in this arena, but none to my knowledge have a similar scope.

Probably the biggest influence on my thinking for this library is the [pbdR package ecosystem](https://github.com/RBigData) for HPC with the R language, which I have worked on for many years now. Some obvious parallels are:

* [float](https://github.com/wrathematics/float) - CPU/GPU
* [kazaam](https://github.com/RBigData/kazaam) - PAR
* [pbdDMAT](https://github.com/RBigData/pbdDMAT) - MPI

The basic philosophy of fml is:

* Be relatively small and self-contained.
* Follow general C++ conventions by default (like RAII and exceptions), but give the ability to break these for the performance-minded.
* Changing a code from one object type to another should be very simple, ideally with no changes to the source (the internals will simply **Do The Right Thing (tm)**), with the exception of:
    - object creation
    - printing
* Use a permissive open source license.
