-include ../make.inc
-include ../common.inc

MPICXX = mpicxx
OMPI_CXX = $(CXX)


all: parmat

SRCS=$(wildcard *.cpp)
OBJS=$(SRCS:.cpp=.o )

%.o: %.cpp 
	$(MPICXX) $(CXXFLAGS) $(CPPFLAGS) $(WARNFLAGS) $(OMPFLAGS) -c $< -o $@

parmat: $(OBJS)
	$(MPICXX) $(CXXFLAGS) $(OBJS) -o parmat $(OMPFLAGS) $(CPU_LDFLAGS)

miniclean:
	( ls *.o | grep -v test-main | xargs rm -f )

run: parmat
	( mpirun -np 2 ./parmat )

clean:
	rm -rf *.o parmat
