#include <stdio.h>
#include <nmmintrin.h>
#include <string.h>
#include <stdint.h>

#include "timer.h"

static uint32_t CrcHash(const void* data, uint32_t bytes);

int main ()
{
    unsigned char data[] = "Adiantum viridimontanum, commonly known as Green Mountain maidenhair fern, is a rare fern found only in outcrops of serpentine rock in New England and Canada. It is named after the site of its discovery in the Green Mountains in Vermont; it has since been located in Quebec and in one site on serpentine in coastal Maine. Until 1991, it was grouped with the western maidenhair fern A. aleuticum, which itself was classified as a variety of the northern maidenhair fern A. pedatum. It was then established that A. viridimontanum was a hybrid species and that the other two ferns were distinct species, although it is difficult to distinguish between the three species in the field. Due to the limited distribution of A. viridimontanum and its similarity to other species, little is known of its ecology. It thrives on sunny, disturbed areas where ultramafic rock is covered with thin soil, such as road cuts, talus slopes, and asbestos mines. Individual plants seem long-lived, and new individuals only infrequently reach maturity.";

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
