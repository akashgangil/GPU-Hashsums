#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <omp.h>

#include "crc.h"
#include "timer.h"
#include "parallel_crc.h"

#define INPUT_FILE "input.in"
#define INPUT_BLOCKS 1
#define DATA_SIZE 1000000
#define BLOCK_SIZE DATA_SIZE/INPUT_BLOCKS


const char* input_data;

void main(void)
{
    char* input_data = malloc(DATA_SIZE);

    FILE* input_file; 
    input_file = fopen(INPUT_FILE, "r");

    if(input_file == NULL){
        fprintf(stderr, "Failed to open %s\n", INPUT_FILE);
        exit(1);
    }

    int i = 0;

    char buffer[150];
    while(fgets(buffer,10,input_file)){
        strcpy(input_data+i,buffer);
        i = i+9;
    }

    /*Replace the line field ascii with \0*/
    input_data[strlen(input_data) - 1] = '\0';
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

    unsigned long ans = crcSlow(input_data1, input_block_size);

    int result = 0;

    int loop_counter = 0;

    for(i = 1; i < INPUT_BLOCKS; ++i){
        input_data1 = malloc(input_block_size + 1);
        strncpy(input_data1, input_data + input_block_size, input_block_size);
        strcat(input_data1, "\0");

        result = crcSlow(input_data1, input_block_size);

        ans = crc32_combine(ans, result, input_block_size);

        loop_counter++;
    }

    printf("LOOP COUNTER    %d\n", loop_counter);

    stopwatch_stop(sw);

    printf("Combined CRC 0x%lx   Time:  %Lg \n",ans, stopwatch_elapsed(sw));

    crcInit();
    stopwatch_start(sw);
    printf("crcFast() 0x%X  ", crcFast(input_data, strlen(input_data)));
    stopwatch_stop(sw);
    printf("  Time: %Lg\n", stopwatch_elapsed(sw));

    stopwatch_destroy(sw);
    free(input_data);
} 
