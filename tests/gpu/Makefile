-include ../make.inc

NVCC = nvcc
NVCCFLAGS = -arch=sm_61

CPPFLAGS = -I../../src
WARNFLAGS = -Wall -Wextra


all: gpumat

SRCS = $(wildcard *.cu)
OBJS = $(SRCS:.cu=.o)

%.o: %.cu 
	$(NVCC) $(NVCCFLAGS) -Xcompiler "$(CXXFLAGS) $(CPPFLAGS) $(WARNFLAGS) $(OMPFLAGS)" -c $< -o $@

gpumat: $(OBJS)
	$(NVCC) test-main.o card.o gpumat.o linalg.o -o gpumat $(GPU_LDFLAGS) -Xcompiler "$(OMPFLAGS)"

miniclean:
	rm -rf gpumat card.o gpumat.o linalg.o

clean:
	rm -rf *.o gpumat
