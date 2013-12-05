
CXX = gcc

all: crc 

crc: crc.o timer.o parallel_crc.o
	$(CXX) -fopenmp -o crc main.c crc.o timer.o parallel_crc.o
	
parallel_crc.o: parallel_crc.h
		$(CXX) -c parallel_crc.c

crc.o: crc.h
	$(CXX) -c crc.c

timer.o: timer.h

clean:
	rm *.o crc 
