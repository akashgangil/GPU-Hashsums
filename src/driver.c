#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "crc_gpu.h"

#define INPUT_FILE "input.in"
#define DATA_SIZE 1000000000
#define BLOCK_SIZE 10

int main (int argc, char** argv){

	// Declare variables
		char* d_A;
		char* combinedCRC = malloc(4); 	//allocate 4 bytes (32 bits) for CRC Checksum
		char* input_data = malloc(DATA_SIZE);
		int i = 0;

	// Loading and processing the input data from file 
		FILE* input_file;
		input_file = fopen(INPUT_FILE, "r");

		if(input_file == NULL){
			fprintf(stderr, "Failed to open %s\n", INPUT_FILE);
			exit(1);
		}
		
		char buffer[BLOCK_SIZE];
		while(fgets(buffer, BLOCK_SIZE, input_file)){
			strcpy(input_data+i,buffer);
			i = i + BLOCK_SIZE - 1;
		}
		fclose(input_file);

		/*Replace the line field ascii with \0*/
		input_data[strlen(input_data) - 1] = '\0';

		
	//Calling function to copy the input data ot the device memory
    		initCudaArray (&d_A, input_data, strlen(input_data) + 1);
	
    
	//Calculate CRC - call the kernel in the function below:
	    	cudaCRC (d_A, strlen(input_data) + 1, combinedCRC);
		fprintf (stderr, "Program Ended Successfully!\n");

	//Free up variables 
	    	free (input_data);

	return 0;
}