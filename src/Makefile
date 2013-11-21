
CXX = gcc

all: crc 

crc: crc.o 
	$(CXX) -o crc main.c crc.o 
	
crc.o: crc.c
	$(CXX) -c crc.c

timer.o: timer.h

clean:
	rm *.o crc
