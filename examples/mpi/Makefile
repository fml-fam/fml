MPICXX = mpicxx
CXXFLAGS = -I../../src
OMPFLAGS = -fopenmp 
MPI_LDFLAGS = -lscalapack-openmpi

WARNFLAGS = -Wall -pedantic -Wextra


all: clean convert matmult mpimat svd


convert:
	$(MPICXX) $(CXXFLAGS) $(WARNFLAGS) $(OMPFLAGS) convert.cpp -o convert $(MPI_LDFLAGS)

matmult:
	$(MPICXX) $(CXXFLAGS) $(WARNFLAGS) $(OMPFLAGS) matmult.cpp -o matmult $(MPI_LDFLAGS)

mpimat:
	$(MPICXX) $(CXXFLAGS) $(WARNFLAGS) $(OMPFLAGS) mpimat.cpp -o mpimat $(MPI_LDFLAGS)

svd:
	$(MPICXX) $(CXXFLAGS) $(WARNFLAGS) $(OMPFLAGS) svd.cpp -o svd $(MPI_LDFLAGS)


clean:
	rm -rf convert matmult mpimat svd
