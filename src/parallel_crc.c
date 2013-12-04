#include<stdio.h>
 
#define GF2_DIM 32

#include "parallel_crc.h"

unsigned long gf2_matrix_times(unsigned long* mat, unsigned long vec){
    unsigned long sum;

    sum = 0;
    while (vec) {
        if (vec & 1)
            sum ^= *mat;
        vec >>= 1;
        mat++;
    }
    return sum;
}

void gf2_matrix_square(unsigned long* square, unsigned long* mat){

    int n;

    for (n = 0; n < GF2_DIM; n++)
        square[n] = gf2_matrix_times(mat, mat[n]);
}

unsigned long crc32_combine(unsigned long crc1, unsigned long crc2, unsigned long len2){

    int n;
    unsigned long row;
    unsigned long even[GF2_DIM];    
    unsigned long odd[GF2_DIM];     

    if (len2 <= 0)
        return crc1;
 
    odd[0] = 0xedb88320UL;      

    row = 1;
    for (n = 1; n < GF2_DIM; n++) {
        odd[n] = row;
        row <<= 1;
    }

    gf2_matrix_square(even, odd);
    gf2_matrix_square(odd, even);

    do {
        gf2_matrix_square(even, odd);
        if (len2 & 1)
            crc1 = gf2_matrix_times(even, crc1);
        len2 >>= 1;

        if (len2 == 0)
            break;

        gf2_matrix_square(odd, even);
        if (len2 & 1)
            crc1 = gf2_matrix_times(odd, crc1);
        len2 >>= 1;

    } while (len2 != 0);

    crc1 ^= crc2;
    return crc1;
}
