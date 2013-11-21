#ifndef _crc_h
#define _crc_h


#define FALSE	0
#define TRUE	!FALSE

//Select the CRC standard from the list that follows.
#define CRC32_MPEG

typedef unsigned long  crc;

#if defined(CRC32_MPEG)


#define CRC_NAME			"CRC32_MPEG"
#define POLYNOMIAL			0x04C11DB7
#define INITIAL_REMAINDER	0xFFFFFFFF
#define FINAL_XOR_VALUE		0xFFFFFFFF
#define REFLECT_DATA		TRUE
#define REFLECT_REMAINDER	TRUE
#define CHECK_VALUE			0xCBF43926

#elif defined(CRC32_SCSI)

#define CRC_NAME			"CRC32_SCSI"
#define POLYNOMIAL		0x1EDC6F41	
#define INITIAL_REMAINDER	0xFFFFFFFF
#define FINAL_XOR_VALUE		0xFFFFFFFF
#define REFLECT_DATA		TRUE
#define REFLECT_REMAINDER	TRUE
#define CHECK_VALUE			0xCBF43926

#else

#error "One of CRC32_SCSI or CRC32_MPEG must be #define'd."

#endif


void  crcInit(void);
crc   crcSlow(unsigned char const message[], int nBytes);
crc   crcFast(unsigned char const message[], int nBytes);


#endif 
