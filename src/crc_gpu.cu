#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "cuda_utils.h"
#include "crc_gpu.h"
#include "parallel_crc.h"

#define BS 1024
#define WIDTH    	(8 * sizeof(int))
#define TOPBIT   	(1 << (WIDTH - 1))

#define FALSE	0
#define TRUE	!FALSE

#define CRC_NAME			"CRC32_MPEG"
#define POLYNOMIAL			0x04C11DB7
#define INITIAL_REMAINDER		0xFFFFFFFF
#define FINAL_XOR_VALUE		0xFFFFFFFF
#define REFLECT_DATA			TRUE
#define REFLECT_REMAINDER		TRUE
#define CHECK_VALUE			0xCBF43926

#if (REFLECT_DATA == TRUE)
#undef  REFLECT_DATA
#define REFLECT_DATA(X)			((unsigned char) reflect((X), 8))
#else
#undef  REFLECT_DATA
#define REFLECT_DATA(X)			(X)
#endif

#if (REFLECT_REMAINDER == TRUE)
#undef  REFLECT_REMAINDER
#define REFLECT_REMAINDER(X)	((int) reflect((X), WIDTH))
#else
#undef  REFLECT_REMAINDER
#define REFLECT_REMAINDER(X)	(X)
#endif

#define GF2_DIM 32

__host__ __device__ unsigned long gf2_matrix_times(unsigned long* mat, unsigned long vec){
    unsigned long sum;

    sum = 0;
    while (vec) {
        if (vec & 1)
            sum ^= *mat;
        vec >>= 1;
        mat++;
    }
    return sum;
}

__host__ __device__ void gf2_matrix_square(unsigned long* square, unsigned long* mat){

    int n;

    for (n = 0; n < GF2_DIM; n++)
        square[n] = gf2_matrix_times(mat, mat[n]);
}

__host__ __device__ unsigned long crc32_combine(unsigned long crc1, unsigned long crc2, unsigned long len2){

    int n;
    unsigned long row;
    unsigned long even[GF2_DIM];    
    unsigned long odd[GF2_DIM];     

    if (len2 <= 0)
        return crc1;
 
    odd[0] = 0xedb88320UL;      

    row = 1;
    for (n = 1; n < GF2_DIM; n++) {
        odd[n] = row;
        row <<= 1;
    }

    gf2_matrix_square(even, odd);
    gf2_matrix_square(odd, even);

    do {
        gf2_matrix_square(even, odd);
        if (len2 & 1)
            crc1 = gf2_matrix_times(even, crc1);
        len2 >>= 1;

        if (len2 == 0)
            break;

        gf2_matrix_square(odd, even);
        if (len2 & 1)
            crc1 = gf2_matrix_times(odd, crc1);
        len2 >>= 1;

    } while (len2 != 0);

    crc1 ^= crc2;
    return crc1;
}




void initCudaArray (char **d_data, char* input_data, unsigned int N)
{
	CUDA_CHECK_ERROR (cudaMalloc ((void**) d_data, N * sizeof (char)));
	CUDA_CHECK_ERROR (cudaMemcpy (*d_data, input_data, N * sizeof (char), cudaMemcpyHostToDevice));

	// In the above two functions, we have copied the value of the input data to the device memory
	// At this point in time after the above 2 functions, device mem has the input data and d_data points to it 

}

__device__ unsigned long reflect(unsigned long data, unsigned char nBits)
{
	unsigned long  reflection = 0x00000000;
	unsigned char  bit;

	//Reflect the data about the center bit.
	for (bit = 0; bit < nBits; ++bit)
	{
		//If the LSB bit is set, set the reflection of it.
		if (data & 0x01)
		{
			reflection |= (1 << ((nBits - 1) - bit));
		}

		data = (data >> 1);
	}

	return (reflection);

}	

//Compute the CRC of a given message.
__device__ int crcSlow(unsigned char const message[], int nBytes)
{
  	int            remainder = INITIAL_REMAINDER;
	int            byte;
	unsigned char  bit;

    //Perform modulo-2 division, a byte at a time.
    for (byte = 0; byte < nBytes; ++byte)
    {
        //Bring the next byte into the remainder.
        remainder ^= (REFLECT_DATA(message[byte]) << (WIDTH - 8));

        //Perform modulo-2 division, a bit at a time.
        for (bit = 8; bit > 0; --bit)
        {
            //Try to divide the current data bit.
            if (remainder & TOPBIT)
            {
                remainder = (remainder << 1) ^ POLYNOMIAL;
            }
            else
            {
                remainder = (remainder << 1);
            }
        }
    }

    //The final remainder is the CRC result.
    return (REFLECT_REMAINDER(remainder) ^ FINAL_XOR_VALUE);
}   


__device__ int crcTable[256];

//Populate the partial CRC lookup table.
__device__ void crcInit(void)
{
	int			   remainder;
 	int			   dividend;
	unsigned char  bit;


    //Compute the remainder of each possible dividend.
    for (dividend = 0; dividend < 256; ++dividend)
    {
        //Start with the dividend followed by zeros.
        remainder = dividend << (WIDTH - 8);

        // Perform modulo-2 division, a bit at a time.
        for (bit = 8; bit > 0; --bit)
        {
            //Try to divide the current data bit.
            if (remainder & TOPBIT)
            {
                remainder = (remainder << 1) ^ POLYNOMIAL;
            }
            else
            {
                remainder = (remainder << 1);
            }
        }

        //Store the result into the table.
        crcTable[dividend] = remainder;
    }

}   


//Compute the CRC of a given message using table lookups 
__device__ int crcFast(char* message, int nBytes)
{
    int	     remainder = INITIAL_REMAINDER;
    unsigned char  data;
    int            byte;


    //Divide the message by the polynomial, a byte at a time.
    for (byte = 0; byte < nBytes; ++byte)
    {
        data = REFLECT_DATA(message[byte]) ^ (remainder >> (WIDTH - 8));
  	 remainder = crcTable[data] ^ (remainder << 8);
    }

    //The final remainder is the CRC.
    return (REFLECT_REMAINDER(remainder) ^ FINAL_XOR_VALUE);

}  


__global__ void crcCalKernel (char* pointerToData, long* partialcrc, unsigned int N)
{

	//Shared memory is per block, so its shared among 1024 threads, as defined by BS -> assume only 16kb of shared mem as what online says 

	__shared__ unsigned long threadlocalcrc[BS];		//This is to store the individual thread's crc - each crc is 4 bytes long (the size of ONE long)
	__shared__ unsigned char buffer[BS * 8];			//This is input data for the entire block - each has 8 bytes to process 
	

	unsigned int i;
	unsigned int numBytesToProcess = 8; 
	unsigned int nThreads = N / numBytesToProcess; 
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x; 
	unsigned int globalStartIndex = threadid * numBytesToProcess; 

/*	//Loading the data into the buffer
	if(threadid < nThreads ){		// each thread copy 8 bytes from global memory 
		for(i = 0; i < numBytesToProcess; i++){	
			if(globalStartIndex + i < N)
				buffer[(threadIdx.x * numBytesToProcess) + i] = pointerToData[globalStartIndex + i];
		}
	}		
	else {
		//do nothing 
	}

	__syncthreads ();

*/


	crcInit();
	if(threadid < nThreads )
	//	threadlocalcrc[threadIdx.x] = (long) crcFast(&buffer[(threadIdx.x * numBytesToProcess)], numBytesToProcess);	//this gives me the crc (as datatype int) for the 8bytes the thread is in charge of 
	
		threadlocalcrc[threadIdx.x] = (long) crcFast(&pointerToData[(globalStartIndex)], numBytesToProcess);	//this gives me the crc (as datatype int) for the 8bytes the thread is in charge of 
	


	//now that data is inside threadlocalcrc, do the combining / reducing 

/*	unsigned int threadOffset = 2; 
	unsigned int partialcrcOffset = 1;
	unsigned int crcSourceSize = numBytesToProcess; 	//16 is the size of data that derived the 2nd crc


	while (threadOffset < nThreads + 1 ){	//plus 1 because we still want the last option where 0 combine with midpoint

		if(threadIdx.x % threadOffset == 0 && threadid < nThreads){
			threadlocalcrc[(threadIdx.x)] = crc32_combine( 	(unsigned long) threadlocalcrc[(threadIdx.x)], 
										(unsigned long) threadlocalcrc[(threadIdx.x) + partialcrcOffset ], 
										crcSourceSize ); 
		}

		threadOffset *= 2;
		partialcrcOffset *= 2;
		crcSourceSize *= 2;

		__syncthreads ();
	}
*/

	

	//Now, all the CRC would be combined into the first element. Add it to the partialcrc variable that was passed in from the host. Only thread 0 does it 
	if(threadIdx.x == 0){

		for(int p = i; p < nThreads; p++){
			threadlocalcrc[(threadIdx.x)] = crc32_combine( 	(unsigned long) threadlocalcrc[(threadIdx.x)], 
										(unsigned long) threadlocalcrc[(threadIdx.x) + p ], 
										8 ); 
	
		}
	

		partialcrc[blockIdx.x] = threadlocalcrc[0];
	}
/*
	long a = 123;
	long b = 456;
	partialcrc[0] = a;
	//partialcrc[1] = ++a;
	//partialcrc[2] = b;
	
*/
	return; 
}




//paramter list: 	1) array pointer value to input data on device global memory 
//			2) number of bytes in input 
//			3) a pointer to CPU memory that will save the combined CRC value

void cudaCRC (char *pointerToData, unsigned int N, char *combinedCRC)	
{

fprintf (stderr, "Entered cuda\n");
	unsigned int	nThreads, tbSize, nBlocks;
	cudaEvent_t start, stop;
	float elapsedTime;	
	long *partialcrc;
	long *hostpartialcrc;
	unsigned long finalcrc; 

	//Determine the number of blocks we need.	
	nThreads = N / 8; 					//Each thread will do a lookup for 8 bytes of N (or 8 characters)
	tbSize = BS;						//The thread block size is limited by the value of BS
	nBlocks = (nThreads + tbSize - 1) / tbSize;	//nBlocks will be the number of blocks we need
	dim3 grid (nBlocks);
	dim3 block (tbSize);

	CUDA_CHECK_ERROR (cudaMalloc ((void**) &partialcrc, sizeof(long) * nBlocks)); 	//create an array to hold the partical CRC from each block, 4 bytes long (i.e. 32 bits)
	hostpartialcrc = (long*) malloc(sizeof(long)* nBlocks);

fprintf (stderr, "nThreads = %u  \n", nThreads);
fprintf (stderr, "tbSize = %u  \n", tbSize);
fprintf (stderr, "nBlocks = %u  \n", nBlocks);
fprintf (stderr, "value in partialcrc = 0X%X  \n", partialcrc);
fprintf (stderr, "value in hostpartialcrc = 0X%X  \n", *hostpartialcrc);
long a = 9;
hostpartialcrc = &a;
fprintf (stderr, "new value of hostpartialcrc = %u  \n", *hostpartialcrc);


	crcCalKernel <<<grid, block>>> (pointerToData, partialcrc, N);
	cudaThreadSynchronize();

	CUDA_CHECK_ERROR (cudaMemcpy (hostpartialcrc, partialcrc, sizeof(long) * nBlocks, cudaMemcpyDeviceToHost));

fprintf (stderr, "After cuda, hostpartialcrc[0] = 0X%X  \n", hostpartialcrc[0]);
//fprintf (stderr, "After cuda, hostpartialcrc[1] = %u  \n", hostpartialcrc[1]);
//fprintf (stderr, "After cuda, hostpartialcrc[2] = %u  \n", hostpartialcrc[2]);

	CUDA_CHECK_ERROR (cudaFree (pointerToData)); 

 
} 





