# fml

* **Version:** 0.1-0
* **License:** [BSD 2-Clause](http://opensource.org/licenses/BSD-2-Clause)
* **Project home**: https://github.com/wrathematics/fml
* **Bug reports**: https://github.com/wrathematics/fml/issues
* **Author:** Drew Schmidt


fml is the Fused Matrix Library, a multi-source, header-only C++ library for matrix computing.

The library provides 4 main classes: `cpumat`, `gpumat`, `parmat`, and `mpimat`. These are mostly what they sound like, but the particular details are:

* CPU: Single node cpu computing (multi-threaded if using multi-threaded BLAS and linking with OpenMP).
* GPU: Single gpu computing via CUDA.
* PAR: Multi-node and/or multi-gpu computing via MPI and/or CUDA.
* MPI: Multi-node computing via ScaLAPACK (+gpus if using [SLATE](http://icl.utk.edu/slate/)).

There are some differences in how objects of any particular type are constructed. But the high level APIs are largely the same between the objects. The goal is to be able to quickly create laptop-scale prototypes that are then easily converted into large scale gpu/multi-node/multi-gpu/multi-node+multi-gpu codes.



## Building and Installation

You can find some examples of how to use the library in the `examples/` tree. Right now there is no real build system beyond some ad hoc makefiles; but ad hoc is better than no hoc.

Depending on which class(es) you want to use, here are some general guidelines for using the library in your own project:

* CPU: `cpumat`
    - Compile with your favorite C++ compiler.
    - Link with LAPACK and BLAS (and ideally with OpenMP). I recommend [OpenBLAS](https://github.com/xianyi/OpenBLAS).
* GPU: `gpumat`
    - Compile with `nvcc`.
    - For most functionality, link with libcudart, libcublas, and libcusolver.  Link with libnvidia-ml if using nvml (using this exclusively does not require `nvcc`; an ordinary C++ compiler will do).
* PAR: `parmat`
    - Compile with `mpicxx`.
    - TODO
* MPI: `mpimat`
    - Compile with `mpicxx`.
    - Link with libscalapack.

Check the makefiles in the `examples/` tree if none of that makes sense.



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
* Other than object creation, changing a code from one class to another should be very simple, ideally with no changes to the source (the internals will simply Do The Right Thing (tm)).
* Use a permissive open source license.
* No Boost, under any circumstances whatsoever.
