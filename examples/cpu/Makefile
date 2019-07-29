CXX = g++
CXXFLAGS = -I../../src

WARNFLAGS = -Wall -pedantic -Wextra


all: clean cpumat

cpumat:
	$(CXX) $(CXXFLAGS) $(WARNFLAGS) cpumat.cpp -o cpumat

clean:
	rm -rf cpumat
