#include <stdio.h>
#include <string.h>

#include "crc.h"


void main(void)
{
	unsigned char  test[] = "123";


	//Print the check value for the selected CRC algorithm.
	printf("The check value for the %s standard is 0x%X\n", CRC_NAME, CHECK_VALUE);
	
	
	// Compute the CRC of the test message, slowly.
	printf("The crcSlow() of \"123456789\" is 0x%X\n", crcSlow(test, strlen(test)));
	
	
	//Compute the CRC of the test message, more efficiently.	 
	crcInit();
	printf("The crcFast() of \"123456789\" is 0x%X\n", crcFast(test, strlen(test)));
} 
