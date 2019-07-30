MPICXX = mpicxx
CXXFLAGS = -I../../src
MPI_LDFLAGS = -lscalapack-openmpi

WARNFLAGS = -Wall -pedantic -Wextra


all: clean mpimat matmult

mpimat:
	$(MPICXX) $(CXXFLAGS) $(WARNFLAGS) mpimat.cpp -o mpimat $(MPI_LDFLAGS)

matmult:
	$(MPICXX) $(CXXFLAGS) $(WARNFLAGS) matmult.cpp -o matmult $(MPI_LDFLAGS)

clean:
	rm -rf mpimat matmult
