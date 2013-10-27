#include<stdio.h>
#include<string.h>

#include<unistd.h>

#include "timer.h"

#define QUOTIENT 0x04c11db7

unsigned int wombat(unsigned char *data, int len)
{
    unsigned int        result;
    int                 i,j;
    unsigned char       octet;
    
    result = -1;
    
    for (i=0; i<len; i++)
    {
        octet = *(data++);
        for (j=0; j<8; j++)
        {
            if ((octet >> 7) ^ (result >> 31))
            {
                result = (result << 1) ^ QUOTIENT;
            }
            else
            {
                result = (result << 1);
            }
            octet <<= 1;
        }
    }
    
    return ~result;             /* The complement of the remainder */
}

char* itoa(int val, int base){
    static char buf[32] = {0};
    int i = 31;
    for(; val && i ; --i, val /= base)
        buf[i] = "0123456789abcdef"[val % base];
    return &buf[i+1];
}


int main(void) 
{
    unsigned char* data = (unsigned char*)"10101111100";
    printf("Message: %s   Length:%zd\n", data, strlen((const char*)data));
    printf("CRC    : %s\n", itoa(wombat(data, 11), 2));
    printf("%s\n", itoa(2, 2));
    
    return 0;
}
