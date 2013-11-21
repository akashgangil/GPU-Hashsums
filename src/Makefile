
CXX = gcc

all: crc 

crc: crc.o timer.o
	$(CXX) -o crc main.c crc.o timer.o
	
crc.o: crc.h
	$(CXX) -c crc.c

timer.o: timer.h

clean:
	rm *.o crc
