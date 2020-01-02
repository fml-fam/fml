-include make.inc

CPPFLAGS = -I../src
CXXFLAGS = -std=c++17 -O2 -march=native
OMPFLAGS = -fopenmp 
WARNFLAGS = -Wall -pedantic -Wextra


all: miniclean arraytools run

SRCS=$(wildcard *.cpp)
OBJS=$(SRCS:.cpp=.o )

%.o: %.cpp 
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(WARNFLAGS) $(OMPFLAGS) -c $< -o $@
 
arraytools: $(OBJS)
	$(CXX) $(OBJS) -o arraytools $(OMPFLAGS)

run: arraytools
	( ./arraytools )

miniclean:
	( ls *.o | grep -v test-main | xargs rm -f )

clean:
	rm -rf *.o arraytools
