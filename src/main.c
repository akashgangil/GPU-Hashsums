#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <omp.h>

#include "crc.h"
#include "timer.h"
#include "parallel_crc.h"

#define INPUT_FILE "input.in"
#define DATA_SIZE 1000000
#define BLOCK_SIZE 10

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
    fclose(input_file);

    /*Replace the line field ascii with \0*/
    input_data[strlen(input_data) - 1] = '\0';
    printf("The Input Data is %s\n", input_data);

    struct stopwatch_t* sw = stopwatch_create();

/*--------------------------------------------------------------------------------------*/
    stopwatch_init();
    stopwatch_start(sw);

    printf("crcSlow() 0x%X  ", crcSlow(input_data, strlen(input_data)));

    stopwatch_stop(sw);   

    printf("  Time: %Lg\n", stopwatch_elapsed(sw));
/*--------------------------------------------------------------------------------------*/

    stopwatch_start(sw);

    size_t input_data_len = strlen(input_data);
    
    int input_blocks = input_data_len / BLOCK_SIZE;
    int extra_blocks = 0;
    if(input_data_len % BLOCK_SIZE != 0)
        extra_blocks = 1;

    int total_blocks = input_blocks + extra_blocks;
    int *result = malloc(total_blocks * sizeof(int));
    
    unsigned long ans = 0;

    char* input_data1 = malloc(BLOCK_SIZE + 1);
    for(i = 0; i < input_blocks; ++i){
        strncpy(input_data1, input_data + BLOCK_SIZE*i, BLOCK_SIZE);
        strcat(input_data1, "\0");
        result[i] = crcSlow(input_data1, BLOCK_SIZE);
    }

    int rem = input_data_len % BLOCK_SIZE;
    if(extra_blocks == 1){
        strncpy(input_data1, input_data + BLOCK_SIZE*i, rem);
        strcat(input_data1, "\0");
        result[i] = crcSlow(input_data1, rem);
    }

    i=0;
    for(i = 0; i<total_blocks-1; ++i){
        ans = crc32_combine(ans, result[i], BLOCK_SIZE);
    }

    ans = crc32_combine(ans, result[i], rem);

    stopwatch_stop(sw);

    printf("Parallel() 0x%lX   Time:  %Lg \n",ans, stopwatch_elapsed(sw));
/*--------------------------------------------------------------------------------------*/

    crcInit();
    stopwatch_start(sw);
    printf("crcFast() 0x%X  ", crcFast(input_data, strlen(input_data)));
    stopwatch_stop(sw);
    printf("  Time: %Lg\n", stopwatch_elapsed(sw));

    stopwatch_destroy(sw);
    free(input_data);
} 
