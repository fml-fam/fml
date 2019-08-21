all: cpumat_run mpimat_run


cpumat:
	( cd cpu; make )

cpumat_run: cpumat
	( cd cpu; ./cpumat )



mpimat:
	( cd mpi; make )

mpimat_run: mpimat
	( cd mpi; ./mpimat )
