MPICXX = mpicxx
CXXFLAGS = -O2
CPPFLAGS = -I../../src
OMPFLAGS = -fopenmp 
MPI_LDFLAGS = -lscalapack-openmpi

WARNFLAGS = -Wall -pedantic -Wextra


all: mpimat

SRCS=$(wildcard *.cpp)
OBJS=$(SRCS:.cpp=.o )

%.o: %.cpp 
	$(MPICXX) $(CXXFLAGS) $(CPPFLAGS) $(WARNFLAGS) $(OMPFLAGS) -c $< -o $@

mpimat: test-main.o mpimat.o linalg.o
	$(MPICXX) $(CXXFLAGS) test-main.o mpimat.o linalg.o -o mpimat $(OMPFLAGS) $(MPI_LDFLAGS)

miniclean:
	rm -rf linalg.o mpimat.o mpimat

clean:
	rm -rf *.o mpimat
