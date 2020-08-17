CXXFLAGS = -O2 -march=native -std=c++17

CPPFLAGS = -I../../src
WARNFLAGS = -Wall -pedantic -Wextra

OMPFLAGS = -fopenmp 

CPU_LDFLAGS = -llapack -lblas
GPU_LDFLAGS = -lcudart -lcublas -lcusolver -lnvidia-ml -lcurand
MPI_LDFLAGS = -lscalapack-openmpi
