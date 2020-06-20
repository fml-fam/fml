-include ../make.inc


all: clean cpumat det matmult svd

cpumat:
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(WARNFLAGS) cpumat.cpp -o cpumat $(CPU_LDFLAGS)

det:
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(WARNFLAGS) det.cpp -o det $(CPU_LDFLAGS)

matmult:
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(WARNFLAGS) matmult.cpp -o matmult $(CPU_LDFLAGS)

svd:
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(WARNFLAGS) svd.cpp -o svd $(CPU_LDFLAGS)

clean:
	rm -rf cpumat det matmult svd
