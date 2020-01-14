all: allminiclean allmake allrun
allminiclean: cpumat_miniclean mpimat_miniclean
allmake: cpumat_make mpimat_make
allrun: cpumat_run mpimat_run

miniclean: cpumat_miniclean mpimat_miniclean
clean: cpumat_clean mpimat_clean



cpumat: cpumat_miniclean cpumat_make cpumat_run

cpumat_make: cpumat_miniclean
	( cd cpu; make -j )

cpumat_run: cpumat_make
	( cd cpu; ./cpumat )

cpumat_miniclean:
	( cd cpu; make miniclean; )

cpumat_clean:
	( cd cpu; make clean; )



gpumat: gpumat_miniclean gpumat_make gpumat_run

gpumat_make: gpumat_miniclean
	( cd gpu; make -j )

gpumat_run: gpumat_make
	( cd gpu; ./gpumat )

gpumat_miniclean:
	( cd gpu; make miniclean; )

gpumat_clean:
	( cd gpu; make clean; )



mpimat: mpimat_miniclean mpimat_make mpimat_run

mpimat_make: mpimat_miniclean
	( cd mpi; make -j )

mpimat_run: mpimat_make
	( cd mpi; make run )

mpimat_miniclean:
	( cd mpi; make miniclean )

mpimat_clean:
	( cd mpi; make clean )
