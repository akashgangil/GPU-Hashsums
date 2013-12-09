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
__device__ int crcSlow(char* message, int nBytes)
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


__global__ void crcCalKernel ( char* pointerToData, long* partialcrc, unsigned int N )
{
	//Note that shared memory max size is 16kb
	__shared__ unsigned long threadlocalcrc[16];		//This is to store the individual thread's crc - each crc is 4 bytes long (the size of ONE long)
	unsigned long localanswer = 0;

	threadlocalcrc[threadIdx.x] = 0;

	crcInit();
	if( threadIdx.x < 15) {
		localanswer = crcSlow(pointerToData + (threadIdx.x * 64) , 64);	//this gives me the crc (as datatype int) for the 8bytes the thread is in charge of 
		threadlocalcrc[threadIdx.x] = localanswer;
	} else {
		localanswer = crcSlow(pointerToData + (threadIdx.x * 64) , 63);	//this gives me the crc (as datatype int) for the 8bytes the thread is in charge of 
		threadlocalcrc[threadIdx.x] = localanswer;
	}

	__syncthreads ();

	

	//Combine with thread 0
	if(threadIdx.x == 0){
		unsigned long localfinalcopyofcrc = 0;

//		unsigned long a = crcSlow(pointerToData, 256);
//		unsigned long b = crcSlow(pointerToData +256, 256);
//		unsigned long c = crcSlow(pointerToData +512, 256);
//		unsigned long d = crcSlow(pointerToData +768, 255);

		localfinalcopyofcrc = crc32_combine (threadlocalcrc[0],   threadlocalcrc[1], 64);
		localfinalcopyofcrc = crc32_combine (localfinalcopyofcrc, threadlocalcrc[2], 64);
		localfinalcopyofcrc = crc32_combine (localfinalcopyofcrc, threadlocalcrc[3], 64);
		localfinalcopyofcrc = crc32_combine (localfinalcopyofcrc, threadlocalcrc[4], 64);
		localfinalcopyofcrc = crc32_combine (localfinalcopyofcrc, threadlocalcrc[5], 64);
		localfinalcopyofcrc = crc32_combine (localfinalcopyofcrc, threadlocalcrc[6], 64);
		localfinalcopyofcrc = crc32_combine (localfinalcopyofcrc, threadlocalcrc[7], 64);
		localfinalcopyofcrc = crc32_combine (localfinalcopyofcrc, threadlocalcrc[8], 64);
		localfinalcopyofcrc = crc32_combine (localfinalcopyofcrc, threadlocalcrc[9], 64);
		localfinalcopyofcrc = crc32_combine (localfinalcopyofcrc, threadlocalcrc[10], 64);
		localfinalcopyofcrc = crc32_combine (localfinalcopyofcrc, threadlocalcrc[11], 64);
		localfinalcopyofcrc = crc32_combine (localfinalcopyofcrc, threadlocalcrc[12], 64);
		localfinalcopyofcrc = crc32_combine (localfinalcopyofcrc, threadlocalcrc[13], 64);
		localfinalcopyofcrc = crc32_combine (localfinalcopyofcrc, threadlocalcrc[14], 64);
		localfinalcopyofcrc = crc32_combine (localfinalcopyofcrc, threadlocalcrc[15], 63);


		partialcrc[0] = localfinalcopyofcrc; 

	}	

	__syncthreads ();

	return; 
}




//paramter list: 	1) array pointer value to input data on device global memory 
//			2) number of bytes in input 
//			3) a pointer to CPU memory that will save the combined CRC value

void cudaCRC (char *pointerToData, unsigned int N, char *combinedCRC)	
{

	unsigned int	nThreads, tbSize, nBlocks;
	cudaEvent_t start, stop;
	float elapsedTime;	
	long *partialcrc;
	long *hostpartialcrc;

	CUDA_CHECK_ERROR (cudaEventCreate (&start));
	CUDA_CHECK_ERROR (cudaEventCreate (&stop));

	//Determine the number of blocks we need.	
	nThreads = 1024 / 8; 					//Each thread will do a lookup for 8 bytes of N (or 8 characters)
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
fprintf (stderr, "value in hostpartialcrc = 0X%X  \n\n", *hostpartialcrc);

CUDA_CHECK_ERROR (cudaEventRecord (start, 0));
	crcCalKernel <<<1, 16>>> (pointerToData, partialcrc, N);
	cudaThreadSynchronize();
	CUDA_CHECK_ERROR (cudaEventRecord (stop, 0));
	CUDA_CHECK_ERROR (cudaEventSynchronize (stop));
	CUDA_CHECK_ERROR (cudaEventElapsedTime (&elapsedTime, start, stop));

	fprintf (stderr, "Execution time: %f ms\n", elapsedTime);

	CUDA_CHECK_ERROR (cudaEventDestroy (start));
	CUDA_CHECK_ERROR (cudaEventDestroy (stop));



	CUDA_CHECK_ERROR (cudaMemcpy (hostpartialcrc, partialcrc, sizeof(long) * nBlocks, cudaMemcpyDeviceToHost));

fprintf (stderr, "After cuda, hostpartialcrc[0] = 0X%X  \n", hostpartialcrc[0]);
fprintf (stderr, "After cuda, hostpartialcrc[0] = %u  \n", hostpartialcrc[0]);


	CUDA_CHECK_ERROR (cudaFree (pointerToData)); 

 
} 





