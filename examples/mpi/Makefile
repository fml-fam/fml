MPICXX = mpicxx
CXXFLAGS = -I../../src
MPI_LDFLAGS = -lscalapack-openmpi

WARNFLAGS = -Wall -pedantic -Wextra


all: clean mpimat

mpimat:
	$(MPICXX) $(CXXFLAGS) $(WARNFLAGS) mpimat.cpp -o mpimat $(MPI_LDFLAGS)

clean:
	rm -rf mpimat
