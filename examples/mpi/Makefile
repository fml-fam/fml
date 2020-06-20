-include ../make.inc


all: clean convert det matmult mpimat svd


convert:
	$(MPICXX) $(CPPFLAGS) $(CXXFLAGS) $(WARNFLAGS) convert.cpp -o convert $(MPI_LDFLAGS)

det:
	$(MPICXX) $(CPPFLAGS) $(CXXFLAGS) $(WARNFLAGS) det.cpp -o det $(MPI_LDFLAGS)

matmult:
	$(MPICXX) $(CPPFLAGS) $(CXXFLAGS) $(WARNFLAGS) matmult.cpp -o matmult $(MPI_LDFLAGS)

mpimat:
	$(MPICXX) $(CPPFLAGS) $(CXXFLAGS) $(WARNFLAGS) mpimat.cpp -o mpimat $(MPI_LDFLAGS)

svd:
	$(MPICXX) $(CPPFLAGS) $(CXXFLAGS) $(WARNFLAGS) svd.cpp -o svd $(MPI_LDFLAGS)


clean:
	rm -rf convert det matmult mpimat svd
