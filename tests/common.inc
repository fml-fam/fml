CXXFLAGS = -O2 -march=native -std=c++17

OMPFLAGS = -fopenmp 

CPU_LDFLAGS = -llapack -lblas
GPU_LDFLAGS = -lcudart -lcublas -lcusolver -lnvidia-ml
MPI_LDFLAGS = -lscalapack-openmpi
