CC = gcc

NVCC = nvcc
CUDA_PATH = /opt/cuda-4.2/cuda
NVCCFLAGS= -arch=compute_20 -code=sm_20 -I$(CUDA_SDK_PATH)/C/common/inc
CFLAGS = -L$(CUDA_PATH)/lib64 -lcudart

all: crc crc_gpu 

crc_CUSRCS = crc_gpu.cu
crc_CSRCS = driver.c
crc_CUOBJS = $(crc_CUSRCS:.cu=.o__cu)
crc_COBJS = $(crc_CSRCS:.c=.o__c)

crc_gpu: $(crc_CUOBJS) $(crc_COBJS)
	$(CC) $(CFLAGS) $^ -o $@ 

%.o__c: %.c
	$(CC) -o $@ -c $<

%.o__cu: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ -c $< 

crc: crc.o timer.o parallel_crc.o
	$(CC) -fopenmp -O3 -o crc main.c crc.o timer.o parallel_crc.o
	
parallel_crc.o: parallel_crc.h
		$(CC) -c parallel_crc.c

crc.o: crc.h
	$(CC) -c crc.c

timer.o: timer.h
	$(CC) -c timer.c

clean:
	rm *.o crc crc_gpu *.o__cu *.o__c 
