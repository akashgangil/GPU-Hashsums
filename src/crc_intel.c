#include <stdio.h>
#include <nmmintrin.h>
#include <string.h>
#include <stdint.h>


static uint32_t CrcHash(const void* data, uint32_t bytes);

int main ()
{
    unsigned char data[] = "223";
    printf("CRC:  0x%X\n", CrcHash((const void*)data, strlen((const char*) data)));
}

static uint32_t CrcHash(const void* data, uint32_t bytes){
    uint32_t hash = 0xFFFFFFFF;

    printf("Data is %s \n", (char*)data);

    uint32_t words = bytes / sizeof(uint32_t);
    bytes = bytes % sizeof(uint32_t);
  

    //const uint32_t* p = reinterpret_cast<const uint32_t*>(data);
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
