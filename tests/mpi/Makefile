-include ../make.inc
-include ../common.inc

MPICXX = mpicxx
OMPI_CXX = $(CXX)


all: mpimat

SRCS=$(wildcard *.cpp)
OBJS=$(SRCS:.cpp=.o )

%.o: %.cpp 
	$(MPICXX) $(CXXFLAGS) $(CPPFLAGS) $(WARNFLAGS) $(OMPFLAGS) -c $< -o $@

mpimat: $(OBJS)
	$(MPICXX) $(CXXFLAGS) $(OBJS) -o mpimat $(OMPFLAGS) $(MPI_LDFLAGS)

miniclean:
	( ls *.o | grep -v test-main | xargs rm -f )

run: mpimat
	( mpirun -np 2 ./mpimat )

clean:
	rm -rf *.o mpimat
