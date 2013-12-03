#include <stdio.h>
#include <string.h>

#include "crc.h"
#include "timer.h"
#include "parallel_crc.h"

unsigned long crc32_combine(unsigned long crc1, unsigned long crc2, unsigned long len2);


void main(void)
{
	unsigned char  test[] = "123";
  unsigned char  test1[]= "12";
  unsigned char  test2[]= "3";


	//Print the check value for the selected CRC algorithm.
	printf("The check value for the %s standard is 0x%X\n", CRC_NAME, CHECK_VALUE);
	
  struct stopwatch_t* sw = stopwatch_create();
  
  stopwatch_init();
  stopwatch_start(sw);

  int result1 = crcSlow(test1, 2);
  printf("Part 1: %X\n", result1);

  int result2 = crcSlow(test2, 1);
	printf("Part 2: %X\n", result2);
	
  printf("Combined CRC 0x%X\n",crc32_combine(result1, result2, 32));

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
