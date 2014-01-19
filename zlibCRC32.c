/* example.c -- usage example of the zlib compression library
 * Copyright (C) 1995-2006, 2011 Jean-loup Gailly.
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

/* @(#) $Id$ */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "zlib.h"
#include "timer.h"

#define DATA_SIZE 1000000000
#define BLOCK_SIZE 1000000
#define INPUT_FILE "/nethome/agangil3/zlib-1.2.8/input.in"

int main(){

    char* input_data = malloc(DATA_SIZE);

    FILE* input_file;
    input_file = fopen(INPUT_FILE, "r");

    if(input_file == NULL){
        fprintf(stderr, "Failed to open %s\n", INPUT_FILE);
        exit(1);
    }

    int i = 0;

    char buffer[BLOCK_SIZE];
    while(fgets(buffer, BLOCK_SIZE, input_file)){
        strcpy(input_data+i,buffer);
        i = i + BLOCK_SIZE - 1;
    }
    fclose(input_file);

    /*Replace the line field ascii with \0*/
    input_data[strlen(input_data) - 1] = '\0';

//    printf("Data: %s\n", input_data);

    struct stopwatch_t* sw = stopwatch_create();

/*--------------------------------------------------------------------------------------*/
    stopwatch_init();
    stopwatch_start(sw);

    printf("CRC: 0X%lX\n", crc32(0L, input_data, strlen(input_data)));
    
    stopwatch_stop(sw);
    printf("Time: %Lg\n", stopwatch_elapsed(sw));
    
    free(input_data);
}
