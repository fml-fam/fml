MPICXX = mpicxx
CXXFLAGS = -I../../src

WARNFLAGS = -Wall -pedantic -Wextra


all: clean comm_hello comm_jid

comm_hello:
	$(MPICXX) $(CXXFLAGS) $(WARNFLAGS) comm_hello.cpp -o comm_hello

comm_jid:
	$(MPICXX) $(CXXFLAGS) $(WARNFLAGS) comm_jid.cpp -o comm_jid

clean:
	rm -rf comm_hello comm_jid
