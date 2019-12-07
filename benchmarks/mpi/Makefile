MPICXX = mpicxx -std=c++17
CPPFLAGS = -I../ -I../../src
CXXFLAGS = -O2 -march=native
OMPFLAGS = -fopenmp 
MPI_LDFLAGS = -lscalapack-openmpi

WARNFLAGS = -Wall -pedantic -Wextra


all: clean matpow


matpow:
	$(MPICXX) $(CPPFLAGS) $(CXXFLAGS) $(WARNFLAGS) $(OMPFLAGS) matpow.cpp -o matpow $(MPI_LDFLAGS)


clean:
	rm -rf matpow
