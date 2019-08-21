MPICXX = mpicxx
CXXFLAGS = -I../../src -O2
MPI_LDFLAGS = -lscalapack-openmpi

WARNFLAGS = -Wall -pedantic -Wextra


all: mpimat

SRCS=$(wildcard *.cpp)
OBJS=$(SRCS:.cpp=.o )

%.o: %.cpp 
	$(MPICXX) $(CXXFLAGS) $(WARNFLAGS) -c $< -o $@

mpimat: test-main.o mpimat.o linalg.o
	$(MPICXX) test-main.o mpimat.o linalg.o -o mpimat $(MPI_LDFLAGS)
	
clean:
	rm -rf *.o mpimat
