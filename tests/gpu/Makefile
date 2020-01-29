-include ../make.inc
-include ../common.inc

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
	$(NVCC) $(OBJS) -o gpumat $(GPU_LDFLAGS) -Xcompiler "$(OMPFLAGS)"

miniclean:
	( ls *.o | grep -v test-main | xargs rm -f )

clean:
	rm -rf *.o gpumat
