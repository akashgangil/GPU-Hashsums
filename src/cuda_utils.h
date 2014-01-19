/**
  *
  *  CUDA Utilities
  *
  *  These functions are similar to the CUDA_SAFE_CALL
  *  from the CUDA SDK as well as book.h from 
  *
  *  http://developer.nvidia.com/cuda-example-introduction-general-purpose-gpu-programming
  *
  *
  **/


#ifndef __CUDA_UTILS_H__
#define __CUDA_UTILS_H__
#include <stdio.h>

static void cuda_check_error(cudaError_t err, const char *file, int line) {
      if (err != cudaSuccess) {
              printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
                  exit(EXIT_FAILURE);
                    }
}

#define CUDA_CHECK_ERROR( err ) (cuda_check_error( err, __FILE__, __LINE__ ))


#define CHECK_NULL( a ) {if (a == NULL) { \
                                printf( "CHECK_ERROR failed in %s at line %d\n", \
                                                                            __FILE__, __LINE__ ); \
                                exit( EXIT_FAILURE );}}


#endif  // __CUDA_UTILS_H__
