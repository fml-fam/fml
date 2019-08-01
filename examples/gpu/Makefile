CXX = g++
NVCC = nvcc
CXXFLAGS = -I../../src
NVCCFLAGS = -arch=sm_61

WARNFLAGS = -Wall -pedantic -Wextra


all: clean nvml

nvml:
	$(CXX) $(CXXFLAGS) $(WARNFLAGS) -I/usr/local/cuda/include nvml.cpp -o nvml -lnvidia-ml

clean:
	rm -rf nvml
