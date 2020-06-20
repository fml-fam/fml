-include ../make.inc
-include ../common.inc


all: cpumat

SRCS=$(wildcard *.cpp)
OBJS=$(SRCS:.cpp=.o )

%.o: %.cpp 
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(WARNFLAGS) $(OMPFLAGS) -c $< -o $@

cpumat: $(OBJS)
	$(CXX) $(OBJS) -o cpumat $(OMPFLAGS) $(CPU_LDFLAGS)

miniclean:
	( ls *.o | grep -v test-main | xargs rm -f )

clean:
	rm -rf *.o cpumat
