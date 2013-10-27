
CXX = g++
CXXFLAGS = -Wall -g

all: crc32naive1 

crc32naive1: crc32naive1.o timer.o
	$(CXX) $(CXXFLAGS) -o crc32naive1 crc32naive1.o timer.o
	
crc32naive1.o: crc32naive1.cpp timer.h
	$(CXX) $(CXXFLAGS) -c crc32naive1.cpp

timer.o: timer.h

clean:
	rm *.o crc32naive1
