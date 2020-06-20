-include ../make.inc


all: clean comm_hello comm_jid

comm_hello:
	$(MPICXX) $(CPPFLAGS) $(CXXFLAGS) $(WARNFLAGS) comm_hello.cpp -o comm_hello $(PAR_LDFLAGS)

comm_jid:
	$(MPICXX) $(CPPFLAGS) $(CXXFLAGS) $(WARNFLAGS) comm_jid.cpp -o comm_jid $(PAR_LDFLAGS)

clean:
	rm -rf comm_hello comm_jid
