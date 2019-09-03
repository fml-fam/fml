CXX = g++
NVCC = nvcc
CXXFLAGS = -I../ -I../../src
LDFLAGS = -llapack -lblas
NVCCFLAGS = -arch=sm_61
NV_LDFLAGS = -lcudart -lcublas -lcusolver -lnvidia-ml

WARNFLAGS = -Wall -pedantic -Wextra


all: clean matmult

matmult:
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) matmult.cu -o matmult $(NV_LDFLAGS) $(LDFLAGS)

clean:
	rm -rf matmult
