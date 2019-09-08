CXX=clang++
#CXX=g++
CXXFLAGS = -O2 -march=native

OMPFLAGS = -fopenmp 

CPU_LDFLAGS = -llapack -lblas
GPU_LDFLAGS = -lcudart -lcublas -lcusolver -lnvidia-ml
MPI_LDFLAGS = -lscalapack-openmpi
