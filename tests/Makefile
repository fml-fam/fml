all: allminiclean allmake allrun
allminiclean: cpumat_miniclean mpimat_miniclean
allmake: cpumat_make mpimat_make
allrun: cpumat_run mpimat_run



cpumat: cpumat_miniclean cpumat_make cpumat_run

cpumat_make:
	( cd cpu; make )

cpumat_run:
	( cd cpu; ./cpumat )

cpumat_miniclean:
	( cd cpu; make miniclean; )

cpumat_clean:
	( cd cpu; make clean; )



mpimat: mpimat_miniclean mpimat_make mpimat_run

mpimat_make:
	( cd mpi; make )

mpimat_run:
	( cd mpi; ./mpimat )

mpimat_miniclean:
	( cd mpi; make miniclean )

mpimat_clean:
	( cd mpi; make clean )



miniclean: cpumat_miniclean mpimat_miniclean
clean: cpumat_clean mpimat_clean
