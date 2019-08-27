CXX = g++
CXXFLAGS = -I../../src -O2
LDFLAGS = -fopenmp -llapack -lblas

WARNFLAGS = -Wall -pedantic -Wextra


all: clean cpumat det matmult svd

cpumat:
	$(CXX) $(CXXFLAGS) $(WARNFLAGS) cpumat.cpp -o cpumat $(LDFLAGS)

det:
	$(CXX) $(CXXFLAGS) $(WARNFLAGS) det.cpp -o det $(LDFLAGS)

matmult:
	$(CXX) $(CXXFLAGS) $(WARNFLAGS) matmult.cpp -o matmult $(LDFLAGS)

svd:
	$(CXX) $(CXXFLAGS) $(WARNFLAGS) svd.cpp -o svd $(LDFLAGS)

clean:
	rm -rf cpumat det matmult svd
