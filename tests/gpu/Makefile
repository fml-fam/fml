CXX = g++
NVCC = nvcc
CXXFLAGS = -I../../src
OMPFLAGS = -fopenmp
NVCCFLAGS = -arch=sm_61
NV_LDFLAGS = -lcudart -lcublas -lcusolver -lnvidia-ml

WARNFLAGS = -Wall -Wextra


all: gpumat

SRCS = $(wildcard *.cu)
OBJS = $(SRCS:.cu=.o)

%.o: %.cu 
	$(NVCC) $(NVCCFLAGS) -Xcompiler "$(CXXFLAGS) $(WARNFLAGS) $(OMPFLAGS)" -c $< -o $@

gpumat: $(OBJS)
	$(NVCC) test-main.o gpumat.o card.o -o gpumat $(NV_LDFLAGS) -Xcompiler "$(OMPFLAGS)"


miniclean:
	rm -rf gpumat gpumat.o card.o

clean:
	rm -rf *.o gpumat
