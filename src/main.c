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

    int i = 0;

    char buffer[150];
    while(fgets(buffer,10,input_file)){
        strcpy(input_data+i,buffer);
        i = i+9;
    }

    printf("The Input Data is %s\n", input_data);
    fclose(input_file);

    struct stopwatch_t* sw = stopwatch_create();

    stopwatch_init();
    stopwatch_start(sw);

    printf("crcSlow() 0x%X  ", crcSlow(input_data, strlen(input_data)));

    stopwatch_stop(sw);   

    printf("  Time: %Lg\n", stopwatch_elapsed(sw));

    stopwatch_start(sw);

    size_t input_data_len = strlen(input_data);
    size_t input_block_size = input_data_len / INPUT_BLOCKS;

    char* input_data1 = malloc(input_block_size+1);
    strncpy(input_data1, input_data, input_block_size);
    *(input_data1 + input_block_size) = '\0';

    int result1 = crcSlow(input_data1, input_block_size);

    char* input_data2 = malloc(input_block_size+1);
    strncpy(input_data2, input_data + input_block_size, input_block_size);
    strcat(input_data2, "\0");

    int result2 = crcSlow(input_data2, input_block_size);

    unsigned long ans = crc32_combine(result1, result2, input_block_size);

    stopwatch_stop(sw);

    printf("Combined CRC 0x%lx  Time:  %Lg \n",ans, stopwatch_elapsed(sw));

    crcInit();
    stopwatch_start(sw);
    printf("crcFast() 0x%X  ", crcFast(input_data, strlen(input_data)));
    stopwatch_stop(sw);
    printf("  Time: %Lg\n", stopwatch_elapsed(sw));

    stopwatch_destroy(sw);
} 
