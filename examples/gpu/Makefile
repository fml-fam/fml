CXX = g++
NVCC = nvcc
CXXFLAGS = -I../../src
NVCCFLAGS = -arch=sm_61
NV_LDFLAGS = -lcudart -lcublas -lcusolver -lnvidia-ml

WARNFLAGS = -Wall -pedantic -Wextra


all: clean card det gpumat half nvml svd

card:
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) card.cu -o card $(NV_LDFLAGS)

det:
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) det.cu -o det $(NV_LDFLAGS)

gpumat:
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) gpumat.cu -o gpumat $(NV_LDFLAGS)

half:
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) half.cu -o half $(NV_LDFLAGS)

nvml:
	$(CXX) $(CXXFLAGS) $(WARNFLAGS) -I/usr/local/cuda/include nvml.cpp -o nvml -lnvidia-ml

svd:
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) svd.cu -o svd $(NV_LDFLAGS)

clean:
	rm -rf card det gpumat half nvml svd
