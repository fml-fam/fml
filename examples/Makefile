all: allclean allmake
allmake: cpumat_make mpimat_make parmat_make
allrun: cpumat_run mpimat_run parmat_run
allclean: cpumat_clean mpimat_clean parmat_clean



cpumat: cpumat_clean cpumat_make

cpumat_make: cpumat_clean
	( cd cpu; make -j )

cpumat_run: cpumat_make
	( cd cpu; ./cpumat )

cpumat_clean:
	( cd cpu; make clean; )



gpumat: gpumat_clean gpumat_make

gpumat_make: gpumat_clean
	( cd gpu; make -j )

gpumat_run: gpumat_make
	( cd gpu; ./gpumat )

gpumat_clean:
	( cd gpu; make clean; )



mpimat: mpimat_clean mpimat_make

mpimat_make: mpimat_clean
	( cd mpi; make -j )

mpimat_run: mpimat_make
	( cd mpi; make run )

mpimat_clean:
	( cd mpi; make clean )



parmat: parmat_clean parmat_make

parmat_make: parmat_clean
	( cd par; make -j )

parmat_run: parmat_make
	( cd par; make run )

parmat_clean:
	( cd par; make clean )
