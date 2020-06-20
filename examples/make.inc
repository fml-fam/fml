CXX=g++
MPICXX = mpicxx

CPPFLAGS = -I../../src/
CXXFLAGS = -O2 -march=native -std=c++17

WARNFLAGS = -Wall -pedantic -Wextra

CPU_LDFLAGS = -fopenmp -llapack -lblas
GPU_LDFLAGS = -lcudart -lcublas -lcusolver -lnvidia-ml
MPI_LDFLAGS = -fopenmp -lscalapack-openmpi
PAR_LDFLAGS = -fopenmp
