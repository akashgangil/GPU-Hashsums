#include <stdio.h>
#include <nmmintrin.h>
#include <string.h>
#include <stdint.h>

#include "timer.h"

#define LENGTH 64000000

static uint32_t CrcHash(const void* data, uint32_t bytes);

void rand_str(char *dest, size_t length) {
    char charset[] = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    
    while (length-- > 0) {
        size_t index = (double) rand() / RAND_MAX * (sizeof charset - 1);
        *dest++ = charset[index];
    }
    *dest = '\0';
}

int main ()
{
    char *data;
    data = (char*) malloc(LENGTH * sizeof(char));

    rand_str(data, LENGTH);
  
    struct stopwatch_t* sw = stopwatch_create();

    stopwatch_init();
    stopwatch_start(sw);   

    printf("CRC:  0x%X\n", CrcHash((const void*)data, strlen((const char*) data)));

    stopwatch_stop(sw);

    printf("%Lg\n", stopwatch_elapsed(sw));

    stopwatch_destroy(sw);

}

static uint32_t CrcHash(const void* data, uint32_t bytes){
    uint32_t hash = 0xFFFFFFFF;

    uint32_t words = bytes / sizeof(uint32_t);
    bytes = bytes % sizeof(uint32_t);
  
    const uint32_t* p = (const uint32_t*)(data);
    while (words--) {
      hash = _mm_crc32_u32(hash, *p);
      ++p;
    }

    const uint8_t* s = (const uint8_t*)(p);
    while (bytes--) {
      hash = _mm_crc32_u8(hash, *s);
      ++s;
    }
  
    return hash ^ 0xFFFFFFFF;
}
