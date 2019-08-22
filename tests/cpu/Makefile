CXX = g++
CXXFLAGS = -I../../src -O2
OMPFLAGS = -fopenmp 
LDFLAGS = -llapack -lblas

WARNFLAGS = -Wall -pedantic -Wextra


all: cpumat

SRCS=$(wildcard *.cpp)
OBJS=$(SRCS:.cpp=.o )

%.o: %.cpp 
	$(CXX) $(CXXFLAGS) $(WARNFLAGS) $(OMPFLAGS) -c $< -o $@

cpumat: test-main.o cpumat.o linalg.o
	$(CXX) test-main.o cpumat.o linalg.o -o cpumat $(OMPFLAGS) $(LDFLAGS)


miniclean:
	rm -rf cpumat cpumat.o linalg.o

clean:
	rm -rf *.o cpumat
