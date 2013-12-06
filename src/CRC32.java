/*************************************************************************
 *  Compilation:  javac CRC32.java
 *  Execution:    java CRC32 s
 *  
 *  Reads in a string s as a command-line argument, and prints out
 *  its 32 bit Cyclic Redundancy Check (CRC32 or Ethernet / AAL5 or ITU-TSS).
 *
 *  Uses direct table lookup, calculation, and Java library.
 *
 *  % java CRC32 123456789
 *  CRC32 (via table lookup)       = cbf43926
 *  CRC32 (via direct calculation) = cbf43926
 *  CRC32 (via Java's library)     = cbf43926
 *
 *
 *
 *  Uses irreducible polynomial:
 *     1    + x    + x^2  + x^4  + x^5  + x^7  + x^8  +
 *     x^10 + x^11 + x^12 + x^16 + x^22 + x^23 + x^26 
 *
 *  0000 0100 1100 0001 0001 1101 1011 0111                 
 *   0    4    C    1    1    D    B    7
 *
 *  The reverse of this polynomial is
 *
 *   0    2    3    8    8    B    D    E
 *
 *
 *
 *************************************************************************/


public class CRC32 { 

    public static void main(String[] args) { 

       /**************************************************************************
        *  Using Java's java.util.zip.CRC32 library
        **************************************************************************/
        java.util.zip.CRC32 x = new java.util.zip.CRC32();
        x.update(bytes);
        System.out.println("CRC32 (via Java's library)     = " + Long.toHexString(x.getValue()));

    }

}

