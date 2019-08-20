CXX = g++
CXXFLAGS = -I../../src -O2
LDFLAGS = -fopenmp -llapack -lblas

WARNFLAGS = -Wall -pedantic -Wextra


all: clean cpumat matmult det

cpumat:
	$(CXX) $(CXXFLAGS) $(WARNFLAGS) cpumat.cpp -o cpumat $(LDFLAGS)
	
matmult:
	$(CXX) $(CXXFLAGS) $(WARNFLAGS) matmult.cpp -o matmult $(LDFLAGS)

det:
	$(CXX) $(CXXFLAGS) $(WARNFLAGS) det.cpp -o det $(LDFLAGS)

clean:
	rm -rf cpumat matmult det
