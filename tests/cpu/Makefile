-include ../make.inc

CPPFLAGS = -I../../src
WARNFLAGS = -Wall -pedantic -Wextra


all: cpumat

SRCS=$(wildcard *.cpp)
OBJS=$(SRCS:.cpp=.o )

%.o: %.cpp 
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(WARNFLAGS) $(OMPFLAGS) -c $< -o $@

cpumat: $(OBJS)
	$(CXX) test-main.o cpumat.o linalg.o -o cpumat $(OMPFLAGS) $(CPU_LDFLAGS)

miniclean:
	rm -rf cpumat cpumat.o linalg.o

clean:
	rm -rf *.o cpumat
