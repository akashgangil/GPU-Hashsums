#include <stdio.h>
#include <string.h>

#include "crc.h"
#include "timer.h"

void main(void)
{
	unsigned char  test[] = "123";


	//Print the check value for the selected CRC algorithm.
	printf("The check value for the %s standard is 0x%X\n", CRC_NAME, CHECK_VALUE);
	
  struct stopwatch_t* sw = stopwatch_create();
  
  stopwatch_init();
  stopwatch_start(sw);

	
	// Compute the CRC of the test message, slowly.
	printf("The crcSlow() of %s is 0x%X\n", test, crcSlow(test, strlen(test)));
  
  stopwatch_stop(sw);   
	
  printf("%Lg\n", stopwatch_elapsed(sw));
  
  
  //Compute the CRC of the test message, more efficiently.	 
	crcInit();
  stopwatch_start(sw);
	printf("The crcFast()  (Table Lookup) of %s is 0x%X\n", test, crcFast(test, strlen(test)));
  stopwatch_stop(sw);
  printf("%Lg\n", stopwatch_elapsed(sw));
  
  stopwatch_destroy(sw);
} 
