#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "crc_gpu.h"

#define INPUT_FILE "input.in"
#define DATA_SIZE 1000000000
#define BLOCK_SIZE 10

int main (int argc, char** argv){

    /* declare variables */
    char *h_A, *d_A, ans;

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
    printf("The Input Data is %s\n", input_data);

    /*Modify it to input string */
    initCudaArray (&d_A, input_data, strlen(input_data) + 1);

    /* do reduction */
    cudaCRC (d_A, strlen(input_data) + 1, &ans);

    free (input_data);

    return 0;
}

