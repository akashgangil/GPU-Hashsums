#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <omp.h>

#include "crc.h"
#include "timer.h"
#include "parallel_crc.h"

#define INPUT_FILE "input.in"
#define INPUT_BLOCKS 2
#define DATA_SIZE 1000000
#define BLOCK_SIZE DATA_SIZE/INPUT_BLOCKS


const char* input_data;

void main(void)
{
  char* input_data = malloc(DATA_SIZE);
  
    
  FILE* input_file; 
  input_file = fopen(INPUT_FILE, "r");
  fscanf(input_file, "%s", input_data);
  fclose(input_file);

  strcat(input_data, "\0");

  printf("Test is %s\n", input_data);

	//Print the check value for the selected CRC algorithm.
	printf("Standard is %s\n", CRC_NAME);
	
  struct stopwatch_t* sw = stopwatch_create();
  
  stopwatch_init();
  stopwatch_start(sw);

  // Compute the CRC of the test message, slowly.
	printf("The crcSlow() of %s is 0x%X\n", input_data, crcSlow(input_data, strlen(input_data)));
  
  stopwatch_stop(sw);   
	
  printf("%Lg\n", stopwatch_elapsed(sw));
 
  stopwatch_start(sw);

  size_t input_data_len = strlen(input_data);
  size_t input_block_size = input_data_len / INPUT_BLOCKS;

  char* input_data1 = malloc(BLOCK_SIZE+1);
  strncpy(input_data1, input_data, input_block_size);
  strcat(input_data1, "\0");

  printf("Part 1 data is %s\n", input_data1);

  int result1 = crcSlow(input_data1, input_block_size);
  printf("Part 1: %X\n", result1);

  
  char* input_data2 = malloc(BLOCK_SIZE+1);
  strncpy(input_data2, input_data + input_block_size, input_block_size);
  strcat(input_data2, "\0");
  
  printf("Part 2 data is %s\n", input_data2);

  int result2 = crcSlow(input_data2, input_block_size);
	printf("Part 2: %X\n", result2);


  printf("INPUT_BLOCK_SIZE: %d\n", input_block_size);
  unsigned long ans = crc32_combine(result1, result2, input_block_size);

  stopwatch_stop(sw);

  printf("Combined CRC 0x%lx     %Lg \n",ans, stopwatch_elapsed(sw));
  
  //Compute the CRC of the test message, more efficiently.	 
	crcInit();
  stopwatch_start(sw);
	printf("The crcFast()  (Table Lookup) of %s is 0x%X\n", input_data, crcFast(input_data, strlen(input_data)));
  stopwatch_stop(sw);
  printf("%Lg\n", stopwatch_elapsed(sw));
  
  stopwatch_destroy(sw);
} 
