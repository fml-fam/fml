SUBDIRS := $(wildcard */.)


all: miniclean cpumat_run mpimat_run

$(SUBDIRS):
	$(MAKE) -C $@


cpumat:
	( cd cpu; make )

cpumat_run: cpumat
	( cd cpu; ./cpumat )


mpimat:
	( cd mpi; make )

mpimat_run: mpimat
	( cd mpi; ./mpimat )


miniclean:
	( cd cpu; make miniclean; cd ..; cd mpi; make miniclean )

clean:
	( cd cpu; make clean; cd ..; cd mpi; make clean )
