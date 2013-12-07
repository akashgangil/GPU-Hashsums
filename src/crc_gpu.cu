#include <stdio.h>

#include "cuda_utils.h"
#include "crc_gpu.h"

#define BS 1024


void initCudaArray (char **d_A, char *h_A, unsigned int N)
{
  CUDA_CHECK_ERROR (cudaMalloc ((void**) d_A, N * sizeof (char)));
  CUDA_CHECK_ERROR (cudaMemcpy (*d_A, h_A, N * sizeof (char), cudaMemcpyHostToDevice));
}

__global__ void crcCalKernel (char* In, char *Out, unsigned int N)
{
  __shared__ char buffer[BS];
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int stride;

  /* load data to buffer */
  if(tid < N) {
    buffer[threadIdx.x] = In[tid];
  } else {
    buffer[threadIdx.x] = (char) 0.0;
  }
  __syncthreads ();

  /* reduce in shared memory */
  for(stride = 1; stride < blockDim.x; stride *= 2) {
    if(threadIdx.x % (stride * 2) == 0) {
      buffer[threadIdx.x] += buffer[threadIdx.x + stride];
    }
    __syncthreads ();
  }

  /* store back the reduced result */
  if(threadIdx.x == 0) {
    Out[blockIdx.x] = buffer[0];
  }
}

char* crcCal (char* d_In, char* d_Out, char* h_Out, unsigned int N)
{
  unsigned int  nThreads, tbSize, nBlocks;
  char* ans;
  

  nThreads = N;
  tbSize = BS;
  nBlocks = (nThreads + tbSize - 1) / tbSize;

  dim3 grid (nBlocks);
  dim3 block (tbSize);

  crcCalKernel <<<grid, block>>> (d_In, d_Out, N);
  cudaThreadSynchronize ();

  CUDA_CHECK_ERROR (cudaMemcpy (h_Out, d_Out, nBlocks * sizeof (char),
                                cudaMemcpyDeviceToHost));

  ans = d_Out;

  return ans;

}

void cudaCRC (char *A, unsigned int N, char *ret)
{
  char *h_Out, *d_Out;
  unsigned int nBlocks;

  cudaEvent_t start, stop;
  float elapsedTime;

  char ans[20];

  nBlocks = (N + BS - 1) / BS;
  h_Out = (char*) malloc (nBlocks * sizeof (char));
  CUDA_CHECK_ERROR (cudaMalloc ((void**) &d_Out, nBlocks * sizeof (char)));
  
  CUDA_CHECK_ERROR (cudaEventCreate (&start));
  CUDA_CHECK_ERROR (cudaEventCreate (&stop));

  fprintf(stderr, "Executing CRC on GPU\n"); 
 
  CUDA_CHECK_ERROR (cudaEventRecord (start, 0));
  /* execute kernel */
  strcpy(ans, crcCal(A, d_Out, h_Out, N)); 
   
  CUDA_CHECK_ERROR (cudaEventRecord (stop, 0));
  CUDA_CHECK_ERROR (cudaEventSynchronize (stop));
  CUDA_CHECK_ERROR (cudaEventElapsedTime (&elapsedTime, start, stop));


  fprintf (stderr, "Execution time: %f ms\n", elapsedTime);

  CUDA_CHECK_ERROR (cudaEventDestroy (start));
  CUDA_CHECK_ERROR (cudaEventDestroy (stop));

  free (h_Out);
  CUDA_CHECK_ERROR (cudaFree (d_Out));

  strcpy(ret, ans); 
}
