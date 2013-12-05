#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <omp.h>

#include "crc.h"
#include "timer.h"
#include "parallel_crc.h"

#define INPUT_FILE "input.in"
#define DATA_SIZE 1000000000
#define BLOCK_SIZE 10

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

    char buffer[10];
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
    
    omp_set_num_threads(8);

    unsigned long ans = 0;

    char* block_data = malloc(input_blocks * (BLOCK_SIZE + 1));
    char* block_addr;
    //#pragma omp parallel  for default(none) shared(input_blocks, input_data, result, block_data) private (i, block_addr)  
    for(i = 0; i < input_blocks; ++i){
        block_addr = block_data + (BLOCK_SIZE + 1) * i;
        strncpy(block_addr, input_data + BLOCK_SIZE * i, BLOCK_SIZE);
        strcat(block_addr, "\0");
        result[i] = crcSlow(block_addr, BLOCK_SIZE);
    }

    int rem = input_data_len % BLOCK_SIZE;

    char* last_block_data = malloc(rem + 1);
    if(extra_blocks == 1){
        strncpy(last_block_data, input_data + BLOCK_SIZE*i, rem);
        strcat(last_block_data, "\0");
        result[i] = crcSlow(last_block_data, rem);
    }

    i=0;
    for(i = 0; i < total_blocks-1; ++i){
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
    free(last_block_data);
    free(block_data);
    free(input_data);
} 
