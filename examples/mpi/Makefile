MPICXX = mpicxx
CXXFLAGS = -I../../src
OMPFLAGS = -fopenmp 
MPI_LDFLAGS = -lscalapack-openmpi

WARNFLAGS = -Wall -pedantic -Wextra


all: clean mpimat matmult mpihelpers

mpimat:
	$(MPICXX) $(CXXFLAGS) $(WARNFLAGS) $(OMPFLAGS) mpimat.cpp -o mpimat $(MPI_LDFLAGS)

matmult:
	$(MPICXX) $(CXXFLAGS) $(WARNFLAGS) $(OMPFLAGS) matmult.cpp -o matmult $(MPI_LDFLAGS)

mpihelpers:
	$(MPICXX) $(CXXFLAGS) $(WARNFLAGS) $(OMPFLAGS) mpihelpers.cpp -o mpihelpers $(MPI_LDFLAGS)

clean:
	rm -rf mpimat matmult mpihelpers
