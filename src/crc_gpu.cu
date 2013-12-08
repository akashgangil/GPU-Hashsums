#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "cuda_utils.h"
#include "crc_gpu.h"

#define BS 1024

void initCudaArray (char **d_data, char* input_data, unsigned int N)
{
	CUDA_CHECK_ERROR (cudaMalloc ((void**) d_data, N * sizeof (char)));
	CUDA_CHECK_ERROR (cudaMemcpy (*d_data, input_data, N * sizeof (char), cudaMemcpyHostToDevice));

	// In the above two functions, we have copied the value of the input data to the device memory
	// At this point in time after the above 2 functions, device mem has the input data and d_data points to it 

}

__global__ void crcCalKernel (char* pointerToData, char* finalcrc, unsigned int N)
{
  
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x; 

  /*__shared__ char buffer[BS];

  // load data to buffer 
  if(tid < N) {
    buffer[threadIdx.x] = In[tid];
  } else {
    buffer[threadIdx.x] = (char) 0.0;
  }
  __syncthreads ();

  / reduce in shared memory 
  for(stride = 1; stride < blockDim.x; stride *= 2) {
    if(threadIdx.x % (stride * 2) == 0) {
      buffer[threadIdx.x] += buffer[threadIdx.x + stride];
    }
    __syncthreads ();
  }

  // store back the reduced result 
  if(threadIdx.x == 0) {
    Out[blockIdx.x] = buffer[0];
  }
*/
}




//paramter list: 	1) array pointer value to input data on device memory 2) number of bytes 
//			3) a pointer to host memory that will save the combined CRC value

void cudaCRC (char *pointerToData, unsigned int N, char *ret)	
{
	unsigned int	nThreads, tbSize, nBlocks;
	cudaEvent_t start, stop;
	float elapsedTime;	
	char* finalcrc;

	//Determine the number of blocks we need.	
	nThreads = N / 8; 					//Each thread will do a lookup of 8 bytes of N
	tbSize = BS;						//The thread block size is limited by the value of BS
	nBlocks = (nThreads + tbSize - 1) / tbSize;	//nBlocks will be the number of blocks we need

	dim3 grid (nBlocks);
	dim3 block (tbSize);

	crcCalKernel <<<grid, block>>> (pointerToData, finalcrc, N);
	cudaThreadSynchronize ();

	//CUDA_CHECK_ERROR (cudaMemcpy (h_Out, d_Out, nBlocks * sizeof (char), cudaMemcpyDeviceToHost));

	CUDA_CHECK_ERROR (cudaFree (pointerToData));

 
}





