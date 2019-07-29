CXX = g++
CXXFLAGS = -I../../src
LDFLAGS = -llapack -lblas

WARNFLAGS = -Wall -pedantic -Wextra


all: clean cpumat matmult

cpumat:
	$(CXX) $(CXXFLAGS) $(WARNFLAGS) cpumat.cpp -o cpumat

matmult:
	$(CXX) $(CXXFLAGS) $(WARNFLAGS) matmult.cpp -o matmult $(LDFLAGS)

clean:
	rm -rf cpumat matmult
