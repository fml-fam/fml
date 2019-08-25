CXX = g++
NVCC = nvcc
CXXFLAGS = -I../../src
NVCCFLAGS = -arch=sm_61
NV_LDFLAGS = -lcudart -lcublas -lcusolver -lnvidia-ml

WARNFLAGS = -Wall -pedantic -Wextra


all: clean card gpumat nvml

card:
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) card.cu -o card $(NV_LDFLAGS)

gpumat:
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) gpumat.cu -o gpumat $(NV_LDFLAGS)

nvml:
	$(CXX) $(CXXFLAGS) $(WARNFLAGS) -I/usr/local/cuda/include nvml.cpp -o nvml -lnvidia-ml

clean:
	rm -rf card gpumat nvml
