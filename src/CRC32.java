/********************************
 * CRC-32 MPEG Java Implementation
 *********************************/


public class CRC32 { 

    public static void main(String[] args) { 

        byte[] bytes = args[0].getBytes();

        java.util.zip.CRC32 x = new java.util.zip.CRC32();
        x.update(bytes);
        System.out.println("CRC32 (via Java's library)     = " + Long.toHexString(x.getValue()));

    }

}

