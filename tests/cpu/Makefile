-include ../make.inc

CPPFLAGS = -I../../src
WARNFLAGS = -Wall -pedantic -Wextra


all: cpumat

SRCS=$(wildcard *.cpp)
OBJS=$(SRCS:.cpp=.o )

%.o: %.cpp 
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(WARNFLAGS) $(OMPFLAGS) -c $< -o $@

cpumat: $(OBJS)
	$(CXX) $(OBJS) -o cpumat $(OMPFLAGS) $(CPU_LDFLAGS)

miniclean:
	rm -rf cpumat cpumat.o linalg.o

clean:
	rm -rf *.o cpumat
